from __future__ import print_function
import multiprocessing
import os
import signal
import subprocess
import sys
import boto3

model_bucket = os.environ.get("MODEL_BUCKET")
model_prefix = os.environ.get("MODEL_PREFIX")
hosting_mode = os.environ.get("HOSTING_MODE", "ECS")

model_dir = '/opt/ml/model'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print(os.listdir(model_dir))

if hosting_mode == "ECS":
    print("Downloading from s3://{}/{}/model.tar.gz to {}".format(model_bucket, model_prefix, model_dir))

    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
    key = "{}/model.tar.gz".format(model_prefix)
    fullpath = os.path.join(model_dir, "model.tar.gz")
    s3.download_file(model_bucket, key, fullpath)

    print("Extracting {} to {}".format(fullpath, model_dir))
    import tarfile
    tar = tarfile.open(fullpath)
    tar.extractall(path=model_dir)
    tar.close()

print("Starting model hosting service...")

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def start_server():
    print('Starting the inference server with {} workers.'.format(model_server_workers))

    # link the log streams to stdout/err so they will be logged to the container logs
    #subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    #subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/program/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')

# The main routine just invokes the start function.


if __name__ == '__main__':
    start_server()
