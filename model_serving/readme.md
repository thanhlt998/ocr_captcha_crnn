# Serving dependencies
* torchserve
* torch-model-archiver

*Note: with conda, torch-model-archiver must be used with python3.8*

# Serving
```
./cmds.sh
```

# Use API
```
curl http://127.0.0.1:8080/predictions/captcha_solver -T imgs/33356.png
```