
# first add Jonathan's repo as the "upstream" remote (you can call upstream whatever you want)
git remote add upstream https://github.com/JGuymont/IFT6390-assignment02.git

# you can see that it worked via:
git remote -v

# now sync the data (in the background)
git fetch upstream

# try to merge the files with your computer
git merge upstream/master

# there's a conflict! we need to deal with this assignment02.pdf 
# file because it isn't in git's brain. so let's do it manually
rm latex/assignment02.pdf 

# finally, voila
git merge upstream/master

# now we can push to our fork
git push origin master
