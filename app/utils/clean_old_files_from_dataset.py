import os
import time

print '83000 initial files'

files_count = 0
for root, dirs, files in os.walk('data'):
    files_count += len(files)

print 'Before deleting we have: {0}'.format(files_count)

for root, dirs, files in os.walk('data'):
    for file in files:
        file_path = os.path.join(root, file)
        created_time = time.ctime(os.path.getctime(file_path))
        if "Sat Dec 10" in created_time:
            os.remove(file_path)


files_count = 0
for root, dirs, files in os.walk('data'):
    files_count += len(files)

print 'After deleting we have: {0}'.format(files_count)
