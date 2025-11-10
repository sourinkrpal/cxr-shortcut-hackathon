### Getting access to the VMs

Send Eike your public ssh key. Typically a file called "id_rsa.pub".

If you already have one, it should be in ~/.ssh (linux) or C:\Users\youraccountname\.ssh (Windows).

If you do *not* have one, you can generate it e.g. using 'ssh-keygen -t ed25519 -C "your_email@example.com"' (Windows).

Place the following ssh config file in ~/.ssh; simply name it "config" (no extension):
```
Host ubra-small-data.mevis.fraunhofer.de
    Port 20353
    User jumper
    ForwardAgent yes

Host workshop
    HostName teamX  # <-- replace X with team number
    ProxyJump ubra-small-data.mevis.fraunhofer.de
    User workshop
```

Teams 1/2 are for this topic (CXR shortcut learning), Teams 3/4 are for SHAP-IQ imputation, Team 5 is for Lipschitz constant estimation.

You should now be able to simply "ssh workshop" from the terminal / powershell.


### Working on the VMs

**Assume that data on the VM can be lost at any time.**

For file synchronization, you can use, e.g., github (for code) or scp (Windows) / rsync (linux). *If using github, be aware that anything you enter on the terminal can be read by your group members and workshop administrators. Do not enter passwords here. As a rule of thumb, do not commit/push to a github repo from the VM (as this requires logging in / authenticating).*

You can either work directly on the terminal (file editors: vim, nano) and/or primarily via file syncing.

You can also directly connect to the VM via SSH within e.g. VisualStudio. (You should still ensure to very regularly sync your data/code elsewhere. *Do not rely on the maintained availability of the VM in any crucial way.*)

*Never forget that other users can see what you type on the command line --> do not enter github passwords, etc.*