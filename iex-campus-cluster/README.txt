To parse IEX data using the Campus Cluster:

1) Run ccsetup.bash. This will transfer all the necessary files to the cluster.

2) ssh into the cluster and run "sbatch ccparse.sbatch". This will start the job. To check the status of the job, run squeue -u [NetID].

3) After the job is complete, download the files using ccdownload.bash. This will download the parsed files from the cluster into book_snapshots.