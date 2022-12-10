data = readContent('dataset.txt');
[train, test] = split_matrix(data, 'un', 0.8);
[i,j,val]=find(sparse(test));
dlmwrite('test.txt',[i j val],'delimiter', ' ','newline','pc');
[i,j,val]=find(sparse(train));
dlmwrite('train.txt',[i j val],'delimiter', ' ','newline','pc');