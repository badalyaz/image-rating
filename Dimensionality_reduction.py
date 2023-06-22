from final_utils import *
from sklearn.decomposition import PCA,KernelPCA,FastICA

# PCA for MG 

#loading feature vectors from paths
root_path = generate_root_path()
paths = glob(f'{root_path}Data/*')
feator_vectors = []
for path in paths:
    feator_vectors.append(np.load(path))
feator_vectors = np.asarray(feator_vectors)
feator_vectors = np.squeeze(feator_vectors,axis = 1)

#creating model PCA and training
pca = PCA(n_components = 8464, kernel = 'rbf' ,eigen_solver = "auto")
pca.fit(feator_vectors)

if not os.path.exists(f'{root_path}Data/mg/original_PCA_8464_auto'):
    os.mkdir(f'{root_path}Data/mg/original_PCA_8464_auto')

pca_path = 'models/PCA/PCA_MG_8464_auto.pkl'
pk.dump(pca, open( pca_path,"wb"))

# Save transformed feature vectors
pca_reload = pk.load(open(pca_path,'rb'))
target = f'{root_path}Data/MG/original_PCA_8464_auto'

for path in paths:
    basename = (os.path.basename(path).split('.'))[0] 
    feat = np.load(path)
    feat = pca_reload.transform(feat)
    np.save(os.path.join(target,basename),feat)

# PCA for CNN
root_path = generate_root_path()
paths = glob(f'{root_path}Data/cnn_efficientnet_b7/border_600x600/*')
feator_vectors = []
for path in paths:
    feator_vectors.append(np.load(path))
print(len(feator_vectors))
feator_vectors = np.asarray(feator_vectors)
feator_vectors = np.squeeze(feator_vectors,axis = 1)
print(feator_vectors.shape)    

#creating model PCA and training
pca_cnn = PCA(n_components = 1280 , svd_solver = "auto")
pca_cnn.fit(feator_vectors)

if not os.path.exists(f'{root_path}Data/cnn_efficientnet_b7/border_600x600_PCA_1280_auto'):
    os.mkdir(f'{root_path}Data/cnn_efficientnet_b7/border_600x600_PCA_1280_auto')

pca_path = 'models/PCA/PCA_CNN_1280_auto.pkl'
pk.dump(pca_cnn, open( pca_path,"wb"))

pca_reload = pk.load(open(pca_path,'rb'))
target =f'{root_path}Data/cnn_efficientnet_b7/border_600x600_PCA_1280_auto'
for path in paths:
    basename = (os.path.basename(path).split('.'))[0] 
    feat = np.load(path)
    feat = pca_reload.transform(feat)
    np.save(os.path.join(target,basename),feat)

# PCA for mg + cnn feature vectors
# feator_vectors = []
# for i in range(len(paths_multigap)):
#     mg_vector = np.load(paths_multigap[i])
#     cnn_vector = np.load(paths_cnn[i])
#     feator_vector = np.concatenate((mg_vector,cnn_vector), axis  = 1)
#     feator_vectors.append(feator_vector)
#     if i == 100:
#         break
#     if i % 1000 == 0:
#         print(i)
# print(len(feator_vectors))
# feator_vectors = np.asarray(feator_vectors)
# feator_vectors = np.squeeze(feator_vectors)
# print(feator_vector



root_path = generate_root_path()

paths_multigap = glob(f'{root_path}Data/MG/original/*')
paths_cnn = glob(f'{root_path}Data/cnn_efficientnet_b7/border_600x600/*')

pca_path = 'models/PCA/mgcnn_pca.pkl'
pca_reload = pk.load(open(pca_path,'rb'))
target =f'{root_path}Data/cnn_mg_concat/pca_9744_auto'

for i in range(len(paths_multigap)):
    basename_mg = (os.path.basename(paths_multigap[i]).split('.'))[0] 
    basename_cnn = (os.path.basename(paths_cnn[i]).split('.'))[0] 
    if basename_cnn == basename_mg:
        feat_mg = np.load(paths_multigap[i])
        feat_cnn = np.load(paths_cnn[i])
        feat = np.concatenate((feat_mg, feat_cnn), axis  = 1)
        feat = pca_reload.transform(feat)
        np.save(os.path.join(target, basename_cnn), feat)
    else:
        print('crush')