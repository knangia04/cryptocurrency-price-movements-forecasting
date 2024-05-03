read -p "Enter your NetID: " username

scp -r $username@cc-login.campuscluster.illinois.edu:./scratch/iex-downloader-parser/data/book_snapshots . 