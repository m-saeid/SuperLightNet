python cls_modelnet.py --embed mlp --batch_size 50
python cls_modelnet.py --embed gaussian --batch_size 50
python cls_modelnet.py --embed fourier --batch_size 50

python cls_scanobject.py --embed mlp --batch_size 50
python cls_scanobject.py --embed gaussian --batch_size 50
python cls_scanobject.py --embed fourier --batch_size 50

python partseg_shapenet.py --embed mlp --batch_size 64
python partseg_shapenet.py --embed gaussian --batch_size 64
python partseg_shapenet.py --embed fourier --batch_size 64











#python semseg_s3dis.py --embed mlp --batch_size 24 --epoch 1



#python cls_modelnet.py --embed mlp --batch_size 50
#python cls_modelnet.py --embed tpe --embed_dim 18 --batch_size 50
#python cls_modelnet.py --embed gpe --batch_size 50
#python cls_modelnet.py --embed pointhop --batch_size 256

#python cls_scanobject.py --embed mlp --batch_size 50
#python cls_scanobject.py --embed tpe --embed_dim 18 --batch_size 50
#python cls_scanobject.py --embed gpe --batch_size 50
#python cls_scanobject.py --embed pointhop --batch_size 256

#python seg_shapenet.py --embed mlp --batch_size 128
#python seg_shapenet.py --embed tpe --embed_dim 36 --batch_size 64
#python seg_shapenet.py --embed gpe --batch_size 128