[train, test] = readData('dataset/',1); % read train file and test file from data directory

[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), train, 'test', test);