for i in {0..50}
do
    python main.py --batch_size 50 --model SLNet_edge0_fps
    
    #python main.py --batch_size 50 --model SLNet_edge25_k3_fps
    #python main.py --batch_size 50 --model SLNet_edge25_k5_fps
    python main.py --batch_size 50 --model SLNet_edge25_k25_fps
    #python main.py --batch_size 50 --model SLNet_edge25_k50_fps

    #python main.py --batch_size 50 --model SLNet_edge50_k3_fps
    #python main.py --batch_size 50 --model SLNet_edge50_k5_fps
    #python main.py --batch_size 50 --model SLNet_edge50_k25_fps
    #python main.py --batch_size 50 --model SLNet_edge50_k75_fps

    #python main.py --batch_size 50 --model SLNet_edge0_fps

    #python main.py --batch_size 50 --model SLNet_edge75_k3_fps
    #python main.py --batch_size 50 --model SLNet_edge75_k5_fps
    #python main.py --batch_size 50 --model SLNet_edge75_k25_fps
    #python main.py --batch_size 50 --model SLNet_edge75_k75_fps

    #python main.py --batch_size 50 --model SLNet_edge100_k3_fps
    #python main.py --batch_size 50 --model SLNet_edge100_k5_fps
    #python main.py --batch_size 50 --model SLNet_edge100_k25_fps
    #python main.py --batch_size 50 --model SLNet_edge100_k50_fps

    #python main.py --batch_size 50 --model SLNet_edge0_fps
done

"""
Models:
SLNet_edge0_fps

SLNet_edge25_k3_fps
SLNet_edge25_k5_fps
SLNet_edge25_k25_fps
SLNet_edge25_k50_fps
SLNet_edge25_k75_fps
SLNet_edge25_k100_fps

SLNet_edge50_k3_fps
SLNet_edge50_k5_fps
SLNet_edge50_k25_fps
SLNet_edge50_k50_fps
SLNet_edge50_k75_fps
SLNet_edge50_k100_fps

SLNet_edge75_k3_fps
SLNet_edge75_k5_fps
SLNet_edge75_k25_fps
SLNet_edge75_k50_fps
SLNet_edge75_k75_fps
SLNet_edge75_k100_fps

SLNet_edge100_k3_fps
SLNet_edge100_k5_fps
SLNet_edge100_k25_fps
SLNet_edge100_k50_fps
SLNet_edge100_k75_fps
SLNet_edge100_k100_fps
"""