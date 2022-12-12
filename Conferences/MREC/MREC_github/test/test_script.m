function test_script(alpha,K,max_iter)
[train, test] = readData('../../MREC_our_interface/',1); % read train file and test file from data directory
train = +(train>0); % convert count into binary, since it is observed that this could lead to higher recommendation performance compared to using count
test = +(test>0); % also convert count into binary

size(train)
alpha = cast(alpha,"double")
K = cast(K,"double")
max_iter = cast(max_iter,"double")
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', alpha, 'K', K, 'max_iter', max_iter), train, 'test', test)
