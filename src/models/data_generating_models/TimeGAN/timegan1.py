import tensorflow as tf
import numpy as np
import pandas as pd
from models.data_generating_models.TimeGAN.utils import extract_time, rnn_cell, random_generator, batch_generator

# tf.compat.v1.disable_eager_execution()

def MinMaxScaler(data):
  """Min-Max Normalizer."""
  min_val = np.min(np.min(data, axis=0), axis=0)
  data = data - min_val
  max_val = np.max(np.max(data, axis=0), axis=0)
  norm_data = data / (max_val + 1e-7)
  return norm_data, min_val, max_val

def embedder(X, T, module_name, hidden_dim, num_layers):
  """Embedding network between original feature space to latent space."""
  with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
    e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    rnn_layer = tf.keras.layers.RNN(e_cell, return_sequences=True)
    H = rnn_layer(X)
    H = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')(H)
  return H

def recovery(H, T, module_name, hidden_dim, dim, num_layers):
  """Recovery network from latent space to original space."""
  with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
    r_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    rnn_layer = tf.keras.layers.RNN(r_cell, return_sequences=True)
    X_tilde = rnn_layer(H)
    X_tilde = tf.keras.layers.Dense(dim, activation='sigmoid')(X_tilde)
  return X_tilde

def generator(Z, T, module_name, hidden_dim, num_layers):
  """Generator function: Generate time-series data in latent space."""
  with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
    g_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    rnn_layer = tf.keras.layers.RNN(g_cell, return_sequences=True)
    E = rnn_layer(Z)
    E = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')(E)
  return E

def supervisor(H, T, module_name, hidden_dim, num_layers):
  """Generate next sequence using the previous sequence."""
  with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
    s_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
    rnn_layer = tf.keras.layers.RNN(s_cell, return_sequences=True)
    S = rnn_layer(H)
    S = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')(S)
  return S

def discriminator(H, T, module_name, hidden_dim, num_layers):
  """Discriminate the original and synthetic time-series data."""
  with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
    d_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
    rnn_layer = tf.keras.layers.RNN(d_cell, return_sequences=True)
    Y_hat = rnn_layer(H)
    Y_hat = tf.keras.layers.Dense(1)(Y_hat)
  return Y_hat

def train_timegan(ori_data, config):
  """Train TimeGAN model and save the parameters."""
  # Clear any existing graph and reset variable scopes
  tf.compat.v1.reset_default_graph()
  
  print(np.asarray(ori_data).shape)
  no, seq_len, dim = np.asarray(ori_data).shape
  ori_time, max_seq_len = extract_time(ori_data)
  ori_data, min_val, max_val = MinMaxScaler(ori_data)

  hidden_dim = config['hidden_dim']
  num_layers = config['num_layer']
  iterations = config['iterations']
  batch_size = config['batch_size']
  module_name = config['module']
  z_dim = dim
  gamma = 1

  # Create a new graph
  graph = tf.Graph()
  with graph.as_default():
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="input_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="input_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="input_t")

    H = embedder(X, T, module_name, hidden_dim, num_layers)
    X_tilde = recovery(H, T, module_name, hidden_dim, dim, num_layers)
    E_hat = generator(Z, T, module_name, hidden_dim, num_layers)
    H_hat = supervisor(E_hat, T, module_name, hidden_dim, num_layers)
    H_hat_supervise = supervisor(H, T, module_name, hidden_dim, num_layers)
    X_hat = recovery(H_hat, T, module_name, hidden_dim, dim, num_layers)
    Y_fake = discriminator(H_hat, T, module_name, hidden_dim, num_layers)
    Y_real = discriminator(H, T, module_name, hidden_dim, num_layers)
    Y_fake_e = discriminator(E_hat, T, module_name, hidden_dim, num_layers)

    e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,:,:], H_hat_supervise[:,:,:])
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V

    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10*tf.sqrt(E_loss_T0)
    E_loss = E_loss0  + 0.1*G_loss_S

    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        if itt % 1000 == 0:
          print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))

      for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        if itt % 1000 == 0:
          print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)))

      for itt in range(iterations):
        for kk in range(2):
          X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
          Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
          _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
          _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if (check_d_loss > 0.15):
          _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        if itt % 1000 == 0:
          print('step: '+ str(itt) + '/' + str(iterations) +
              ', d_loss: ' + str(np.round(step_d_loss,4)) +
              ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) +
              ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) +
              ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) +
              ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
      saver.save(sess, 'data/params/time_gan/timegan_model.ckpt')

def generate_synthetic_data(ori_data, config, M):
  """Generate synthetic data using trained TimeGAN model."""
  # Clear any existing graph and reset variable scopes
  tf.compat.v1.reset_default_graph()
  
  no, seq_len, dim = np.asarray(ori_data).shape
  ori_time, max_seq_len = extract_time(ori_data)
  ori_data, min_val, max_val = MinMaxScaler(ori_data)

  hidden_dim = config['hidden_dim']
  num_layers = config['num_layer']
  module_name = config['module']
  z_dim = dim

  # Create a new graph
  graph = tf.Graph()
  with graph.as_default():
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="input_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="input_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="input_t")

    H = embedder(X, T, module_name, hidden_dim, num_layers)
    X_tilde = recovery(H, T, module_name, hidden_dim, dim, num_layers)
    E_hat = generator(Z, T, module_name, hidden_dim, num_layers)
    H_hat = supervisor(E_hat, T, module_name, hidden_dim, num_layers)
    H_hat_supervise = supervisor(H, T, module_name, hidden_dim, num_layers)
    X_hat = recovery(H_hat, T, module_name, hidden_dim, dim, num_layers)
    Y_fake = discriminator(H_hat, T, module_name, hidden_dim, num_layers)
    Y_real = discriminator(H, T, module_name, hidden_dim, num_layers)
    Y_fake_e = discriminator(E_hat, T, module_name, hidden_dim, num_layers)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(graph=graph) as sess:
      saver.restore(sess, 'data/params/time_gan/timegan_model.ckpt')
      Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
      generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

      generated_data = list()
      for i in range(no):
        temp = generated_data_curr[i,:ori_time[i],:]
        generated_data.append(temp)

      generated_data = generated_data * max_val
      generated_data = generated_data + min_val

  return generated_data



def train_and_generate(ori_data, config, M):
  """Train TimeGAN model and generate synthetic data. Just a combination of the two functions above."""
  # Clear any existing graph and reset variable scopes
  tf.compat.v1.reset_default_graph()
  
  print(np.asarray(ori_data).shape)
  no, seq_len, dim = np.asarray(ori_data).shape
  ori_time, max_seq_len = extract_time(ori_data)
  ori_data, min_val, max_val = MinMaxScaler(ori_data)

  hidden_dim = config['hidden_dim']
  num_layers = config['num_layer']
  iterations = config['iterations']
  batch_size = config['batch_size']
  module_name = config['module']
  z_dim = dim
  gamma = 1

  # Create a new graph
  graph = tf.Graph()
  with graph.as_default():
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="input_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="input_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="input_t")

    H = embedder(X, T, module_name, hidden_dim, num_layers)
    X_tilde = recovery(H, T, module_name, hidden_dim, dim, num_layers)
    E_hat = generator(Z, T, module_name, hidden_dim, num_layers)
    H_hat = supervisor(E_hat, T, module_name, hidden_dim, num_layers)
    H_hat_supervise = supervisor(H, T, module_name, hidden_dim, num_layers)
    X_hat = recovery(H_hat, T, module_name, hidden_dim, dim, num_layers)
    Y_fake = discriminator(H_hat, T, module_name, hidden_dim, num_layers)
    Y_real = discriminator(H, T, module_name, hidden_dim, num_layers)
    Y_fake_e = discriminator(E_hat, T, module_name, hidden_dim, num_layers)

    e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,:,:], H_hat_supervise[:,:,:])
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V

    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10*tf.sqrt(E_loss_T0)
    E_loss = E_loss0  + 0.1*G_loss_S

    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        if itt % 1000 == 0:
          print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))

      for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        if itt % 1000 == 0:
          print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)))

      for itt in range(iterations):
        for kk in range(2):
          X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
          Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
          _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
          _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if (check_d_loss > 0.15):
          _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        if itt % 1000 == 0:
          print('step: '+ str(itt) + '/' + str(iterations) +
              ', d_loss: ' + str(np.round(step_d_loss,4)) +
              ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) +
              ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) +
              ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) +
              ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
          
    with tf.compat.v1.Session(graph=graph) as sess:
      # saver.restore(sess, 'data/params/time_gan/timegan_model.ckpt')
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

        generated_data = list()
        for i in range(no):
          temp = generated_data_curr[i,:ori_time[i],:]
          generated_data.append(temp)

        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

    return generated_data