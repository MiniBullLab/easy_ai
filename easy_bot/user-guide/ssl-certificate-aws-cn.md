# Create Free SSL Certificate and Upload to AWS China Regions

In AWS China regions, the Amazon CloudFront cannot associate with SSL certificate issued by Amazon Certificate Manager (ACM).
In order to protect your custom domain, you need to upload a SSL certificate into AWS Identity & Access Management (IAM). 

You can create a free SSL certificate provided by [Let's Encrypt](https://letsencrypt.org/). In production environment, 
we suggest you to use your own SSL certificate. Please note, this free SSL certificate will expire in 3 months.

## Create a Free SSL Certificate

1. Install the `certbot` CLI. On MacOS, open an terminal and run `brew install certbot`. On Windows, download the 
   latest version of [Certbot installer](https://dl.eff.org/certbot-beta-installer-win32.exe) and follow the wizard.
   
1. On Windows, To start a shell for Certbot, select the Start menu, enter cmd (to run CMD.EXE) or 
   powershell (to run PowerShell), and click on “Run as administrator” in the contextual menu that shows up above.
   
1. Obtain a SSL certificate. Remove `sudo` on Windows platform.
   ```
   sudo certbot certonly --manual --preferred-challenges dns -d "<your-domain-name>"
   ```

1. It will prompt you like the following 
   ```
   Please deploy a DNS TXT record under the name
   <acme-challenge-domain> with the following value:
   
   <TXT-record-value>********
   
   ```
   
1. Create a **TXT** type record in your DNS resolver. Set `<acme-challenge-domain>` points to `<TXT-record-value>`.

1. Press **Enter** button to continue, and you will obtain the certificate.

_Note_: If you get the following error, it means you reached the API limit. You may need to wait or change the local IP. 
If you are in the corp VPN network, try to disconnect and run the command again.

## Upload the Certificate to AWS IAM

1. Follow the instruction to [install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html), 
   and [configure the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Skip this step if you have done before.
   
1. Upload to AWS IAM using CLI.
   ```
   sudo aws iam upload-server-certificate \
   --path `/cloudfront/` \
   --server-certificate-name '<domain-name>' \
   --certificate-body file:///etc/letsencrypt/live/<domain-name>-xxx/cert.pem \
   --private-key file:///etc/letsencrypt/live/<domain-name>-0001/privkey.pem \
   --certificate-chain file:///etc/letsencrypt/live/<domain-name>-0001/chain.pem \
   ```

## Find the ServerCertificateId

1. Use `aws iam list-server-certificate` to list all certificates. 

1. Find the **ServerCertificateId** of uploaded certificate. 

