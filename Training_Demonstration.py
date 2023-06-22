from final_utils import *


# trainer
def trainer(model, data, weights_path, data_val,batch_size=128, epochs=30, learning_rate=0.03):
    X_train, X_test, y_train, y_test = data

    model.compile(loss=SC_CE_KLD,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, 
                                                  epsilon=1e-07, decay=0, amsgrad=False))
    model.load_weights(weights_path) 
    checkpoint = keras.callbacks.ModelCheckpoint(weights_path, 
                                                 monitor='val_loss', 
                                                 verbose=1, 
                                                 save_best_only=True, 
                                                 mode='min')
    schedule = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)
    callbacks_list = [checkpoint, schedule]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data = data_val)

    return history

if __name__ == "__main__":
    # generate root path
    root_path = generate_root_path() 

    # Multigap feature extraction
    # creating model
    model_mg = model_inceptionresnet_multigap()
    # extracting train good image multigap features and saving with .json
    source_file = root_path + 'Data/splitted/train/images/good/good1'
    target_file = root_path + 'Data/splitted/train/features/mg/original/'

    extract_features_from_path_automated_json(
                                        source_file=source_file,
                                        target_file=target_file,
                                        label='good',
                                        splitted='good1',
                                        model=model_mg,
                                        resize_func=False,
                                        save_json=True)

    # extracting train bad image multigap features and saving with .json
    target_file = root_path + 'Data/splitted/train/features/multigap/original/'

    for i in range(7):
        paths = root_path + f'Data/splitted/train/images/bad/bad{i+1}'
        extract_features_from_path_automated_json(
                                        source_file=paths,
                                        target_file=target_file,
                                        label='bad', 
                                        splitted=f'bad{i+1}',
                                        model=model_mg,
                                        resize_func=False,
                                        save_json=True)
        
    # creating model
    model_cnn = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",trainable=False) ])
    # extracting train good image cnn features and saving
    source_file = root_path + 'Data/splitted/train/images/good/good1'
    target_file = root_path + 'Data/splitted/train/features/cnn_efficientnet_b7/border_600x600/'

    extract_features_from_path_automated_json(
                                        source_file=source_file,
                                        target_file=target_file,
                                        label='good',
                                        splitted='good1',
                                        model=model_cnn, 
                                        resize_func=resize_add_border,
                                        size=(600, 600),
                                        save_json=False)

    # extracting train bad image cnn features and saving
    target_file = root_path + 'Data/splitted/train/features/cnn_efficientnet_b7/border_600x600/'

    for i in range(7):
        paths = root_path + f'Data/splitted/train/images/bad/bad{i+1}'
        extract_features_from_path_automated_json(
                                        source_file=paths,
                                        target_file=target_file,
                                        label='bad', 
                                        splitted=f'bad{i+1}',
                                        model=model_cnn, 
                                        resize_func=resize_add_border,
                                        size=(600, 600),
                                        save_json=False)
        
    # loading feature vectors from paths
    paths = glob(f'{root_path}Data/splitted/train/features/mg/original/*')
    feator_vectors = []
    for path in paths:
        feator_vectors.append(np.load(path))
    feator_vectors = np.asarray(feator_vectors)
    feator_vectors = np.squeeze(feator_vectors,axis = 1)
    print(feator_vectors.shape)    

    # repeats need for PCA model fit
    feator_vectors = np.repeat(feator_vectors, 200, axis=0)

    # creating model PCA and training
    pca_mg = PCA(n_components = 8464 , svd_solver = "auto")
    pca_mg.fit(feator_vectors)#

    # making directores for saving transformed features and pca model
    transformed_feats_path = f'{root_path}Data/splitted/train/features/mg/original_PCA_8464_auto'
    pca_mg_path = 'models/PCA'

    if not os.path.exists(transformed_feats_path):
        os.mkdir(transformed_feats_path)
    if not os.path.exists(pca_mg_path):
        os.mkdir(pca_mg_path)

    # saving pca model
    pk.dump(pca_mg, open(f'{pca_mg_path}/PCA_MG_8464_auto.pkl', 'wb'))

    # saving transformed features
    for path in paths:
        basename = (os.path.basename(path).split('.'))[0]
        feat = np.load(path)
        feat = pca_mg.transform(feat)
        np.save(os.path.join(transformed_feats_path, basename), feat)

    # loading feature vectors from paths
    paths = glob(f'{root_path}Data/splitted/train/features/cnn_efficientnet_b7/border_600x600/*')
    feator_vectors = []
    for path in paths:
        feator_vectors.append(np.load(path))
    feator_vectors = np.asarray(feator_vectors)
    feator_vectors = np.squeeze(feator_vectors,axis = 1)
    print(feator_vectors.shape)    

    # repeats need for PCA model fit
    feator_vectors = np.repeat(feator_vectors, 30, axis=0)

    # creating model PCA and training
    pca_cnn = PCA(n_components = 1280 , svd_solver = "auto")
    pca_cnn.fit(feator_vectors)

    # making directores for saving transformed features and pca model
    transformed_feats_path = f'{root_path}Data/splitted/train/features/cnn_efficientnet_b7/border_600x600_PCA_1280_auto'
    pca_cnn_path = 'models/PCA'
    if not os.path.exists(transformed_feats_path):
        os.mkdir(transformed_feats_path)
    if not os.path.exists(pca_cnn_path):
        os.mkdir(pca_cnn_path)

    # saving pca model
    pk.dump(pca_cnn, open(f'{pca_cnn_path}/PCA_CNN_1280_auto.pkl', 'wb'))
    # saving transformed features
    for path in paths:
        basename = (os.path.basename(path).split('.'))[0]
        feat = np.load(path)
        feat = pca_cnn.transform(feat)
        np.save(os.path.join(transformed_feats_path, basename), feat)

    # Training FC
    # define loss function
    SC_CE_KLD = tf.keras.losses.SparseCategoricalCrossentropy()
    # Loading cnn features and multigap features from .json
    main_path = root_path + 'Data/splitted/train/'
    features_bad_list = []
    features_bad_list_i = []
    features_good1 = []
    feats_MG = 'original_PCA_8464_auto' 
    feats_CNN = 'border_600x600_PCA_1280_auto'
        
    for i in range(7):
        alm_train_bad = open(f'{main_path}data_bad{i+1}.json')
        bad_data = json.load(alm_train_bad)
        for data in bad_data:
            feat_path_1 = main_path + f'features/mg/{feats_MG}/' + data['feature']
            feat_path_2 = main_path + f'features/cnn_efficientnet_b7/{feats_CNN}/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
            features_bad_list_i.append(connected)
        features_bad_list.append(features_bad_list_i)
        features_bad_list_i = []
            
    alm_train_good = open(f'{main_path}/data_good1.json')
    good_data = json.load(alm_train_good)
    for data in good_data:
        feat_path_1 = main_path + f'features/mg/{feats_MG}/' + data['feature']
        feat_path_2 = main_path + f'features/cnn_efficientnet_b7/{feats_CNN}/' + data['feature']
        connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load((feat_path_2)))))
        features_good1.append(connected)
        
    for i in range(7):
        features_bad_list[i] = np.squeeze(np.array(features_bad_list[i]))
    features_good1 = np.squeeze(np.array(features_good1))
    
    # Generating static validation data
    features_bad_list[0], features_bad1_val = extract_static_val_data(features_bad_list[0], perc = 0.11)
    features_good1, features_good1_val = extract_static_val_data(features_good1, perc = 0.11)

    bad = features_bad_list
    # Creating validation data
    X_val = np.concatenate((features_bad1_val, features_good1_val) , axis=0 )
    y_val = np.concatenate((np.zeros(len(features_bad1_val)), np.ones(len(features_good1_val))), axis=0 )

    # creating fc model and weights
    model_fc = fc_model_softmax(input_num=9744)
    weights_path = f'models/model_fc_softmax.hdf5' #path where will save model weights
    model_fc.save_weights(weights_path) #if we want to cancel learning and start from 0, if not comment the line
    model_fc.load_weights(weights_path)

    # defining epochs count, batch size and learning rate
    epochs = 15
    batch_size = 128
    learning_rate = 0.003

    data_val = (X_val, y_val)

    i = 20
    data = get_train_pairs(bad[i], features_good1, train_size=0.95, shuffle=True)
    history = trainer(model_fc, data, weights_path, data_val, batch_size, epochs, learning_rate=learning_rate)
    acc = calc_acc(model_fc, weights_path, data_val[0], data_val[1], batch_size)
    print('----- Accuracy =', acc, '%', ' -----')
    print('---Batch Train Done---')