# Keras Model serving with CI and Heroku

Check out the blogpost [here](https://medium.com/p/35648f9dc5fb)

## Build docker image
```
    # Prepare for heroku container deployment
    docker build -t registry.gitlab.com/fast-science/background-removal-server .
    # Or for gitlab
    docker build -t registry.gitlab.com/fast-science/background-removal-server .
```

## Deploying to a docker machine

```
    docker login
    # For plain HTTP
    docker run -t -i -p80:5001 -ePORT=5001 registry.gitlab.com/fast-science/background-removal-server
    # or with https
    docker run -ti -p443:5001 -ePORT=5001 -v"$PWD/certificate:/certificate:ro" --entrypoint "gunicorn --threads 20 --bind 0.0.0.0:$PORT --timeout 120 wsgi --certfile=/certificate/server.crt --keyfile=/certificate/server.key" registry.gitlab.com/fast-science/background-removal-server
    docker run -ti -p443:5001 -v"$PWD/certificate:/certificate:ro" registry.gitlab.com/fast-science/background-removal-server gunicorn --threads 20 --bind 0.0.0.0:5001 --timeout 120 wsgi --certfile=/certificate/server.crt --keyfile=/certificate/server.key
```
