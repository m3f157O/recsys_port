[train, test] = readData('../../MREC_our_interface/',1); % read train file and test file from data directory
train = +(train>0); % convert count into binary, since it is observed that this could lead to higher recommendation performance compared to using count
test = +(test>0); % also convert count into binary

size(train)
size(test)
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), train, 'test', test)