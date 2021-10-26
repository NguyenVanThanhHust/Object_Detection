conda create -n object_detection python=3.7
conda activate object_detection
conda install -y matplotlib scipy
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip3 install Cython
git clone https://github.com/philferriere/cocoapi.git
cd cocoapi && cd PythonAPI && python3 setup.py install