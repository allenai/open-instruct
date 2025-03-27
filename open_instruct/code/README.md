"""
can launch local server with:
```
python -m open_instruct.code.api 
```

or launch the server in a docker container:
```
docker build -t code-api -f open_instruct/code/Dockerfile .
docker run -p 8000:8000 code-api
```
NOTE: at the moment, docker container suddenly stops running after ~30 minutes of training
Still investigating this. not a problem with running it the other way

and then test with:
```
python -m pytest open_instruct/code/test_api.py -v
```
"""