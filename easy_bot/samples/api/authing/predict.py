import json
import requests
import base64
from authing.v2.management import ManagementClient, ManagementClientOptions
from authing.v2.authentication import AuthenticationClient, AuthenticationClientOptions

## setup application in authing
app_id = "6017b44b1bcexxxxxxxx"
email = "xxxxxx@amazon.com"
password = "xxxxxxx"
portalDomain = "ml-bot.xxxxx.cn"

## set task name and test image sample
task_id = "实勘图分类"
img_filename = "test.jpg"

authentication_client = AuthenticationClient(
  options=AuthenticationClientOptions(
    app_id=app_id
))
user = authentication_client.login_by_email(
    email=email,
    password=password,
)
print(json.dumps(user, indent=2))

token = user['token']
# print(token)

response = requests.get("https://{}/aws-exports.json".format(portalDomain))
config = json.loads(response.text)
print(config)
rootUrl = config["apiUrl"]

## attempt to retrieve task list
taskUrl = "{}tasks".format(rootUrl)
headers = {"authorization":"Bearer {}".format(token)}
response = requests.get(taskUrl, headers = headers)

## convert image data to json string
imagedata = ""
with open(img_filename, "rb") as fp:
    imagedata = base64.b64encode(fp.read())
imagedata_hex = "data:image/jpeg;base64,{}".format(imagedata.decode())

## send prediction request
predictUrl = "{}/{task_id}/predict".format(taskUrl, task_id=task_id)
print(f"> {predictUrl}")
data = { "imagedata": imagedata_hex }
response = requests.post(predictUrl, json=data, headers = headers)
print(response.json())

## sample response:
# {'Status': 'Success', 'taskName': '实勘图分类', 'Message': '',
# 'ClassIndex': 1,
# 'Results': [
#   {'Class': '卫生间', 'Probability': 0.4944167733192444},
#   {'Class': '厨房', 'Probability': 0.5055732727050781},
#   {'Class': '客厅', 'Probability': 7.045939128147438e-06}
# ]}
