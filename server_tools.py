# https://stackoverflow.com/questions/432385/sftp-in-python-platform-independent
import paramiko
import sys
import os
assert 'SYSTEMROOT' in os.environ
import socket
""""Currently, it seems useless."""
socket.getaddrinfo('127.0.0.1', 8080)
def download(fn, dst, sftp):
    path = f'/mnt/disk1/youqing/prepared/zips/overview/{fn}'
    localpath = dst
    sftp.get(localpath, path)

def batch_download(fns, dst):
    host = "youqing@219.135.135.229"                    #hard-coded
    port = 22
    transport = paramiko.Transport((host, port))

    password = "22706"                #hard-coded
    username = "youqing"                #hard-coded
    transport.connect(username=username, password=password)

    sftp = paramiko.SFTPClient.from_transport(transport)

    for fn in fns:
        download(fn, dst, sftp)
    sftp.close()
    transport.close()

def get_fns(fns_p):
    with open(fns_p, 'r') as temp:
        fns = list(temp.readlines())
    return fns

if __name__ == "__main__":
    fns_p = sys.argv[1]
    dst = sys.argv[2]
    batch_download(get_fns(fns_p), dst)