read -p "Enter your NetID: " username

scp -r ./iex-downloader-parser/ $username@cc-login.campuscluster.illinois.edu:./scratch