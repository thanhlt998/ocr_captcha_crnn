torch-model-archiver --model-name captcha_solver --version 1.0 --serialized-file captcha_data6.pkl --model-file crnn.py --handler model_handler.py --extra-files captcha_data6.pkl
mkdir model_store
mv *.mar model_store
torchserve --start --ncs --model-store model_store --models captcha_solver=captcha_solver.mar
