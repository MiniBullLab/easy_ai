import boto3
import time
import json

runtime = boto3.client("sagemaker-runtime")

tic = time.time()

body = '{"data":"15381861666牟王斌浙江省台州市黄岩区沙埠镇邱家村23号"}'

response = runtime.invoke_endpoint(
    EndpointName='yellowpage-3',
    Body=body,
    ContentType='application/json',
)
body = response["Body"].read()

toc = time.time()

data_dict = json.loads(body.decode())

print(data_dict)
print(f"elapsed: {(toc - tic) * 1000.0} ms")
