clear all;
clc;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

S1 = 530;
S2 = 10;

target = full(ind2vec(labels+1));

[I, Q] = size(images)
[O, Q] = size(target)

net = newff(minmax(images),[S1 S2], {'logsig','logsig'},'traingd');
net.inputWeights{1,1}.initFcn = 'rands';
net.LW{2,1}.initFcn = 'rands';
net.b{2}.initFcn='rands';
net.b{1}.initFcn='rands';

net.performFcn = 'mse';
net.trainFcn = 'traingd';
net.trainParam.goal = 0;
net.trainParam.show = 1;
net.trainparam.epochs = 60;
%net.trainParam.mc = 0.9;
net.trainParam.lr = 0.3;
net=init(net);
bn = 1;
trn = 1;
for bn = 1:500:10000
    trn = bn
    if(bn == 59995) break
    end
    imagesBATCH = images(:,bn:bn+499);
    targetBATCH = target(:,trn:trn+499);
    [net,tr] = train(net,imagesBATCH,targetBATCH);
end

imagesUSE = loadMNISTImages('t10k-images.idx3-ubyte');
labelsUSE = loadMNISTLabels('t10k-labels.idx1-ubyte');

targetUSE = full(ind2vec(labelsUSE+1));
output = sim(net,imagesUSE)
for index = 1:1:10000
    [M,I] = max(output(:,index))
    I = I - 1
    J = labelsUSE(index)
    if(I == J)
        true = true + 1
    else
        false = false + 1
    end
end

 %output = net(imagesUSE);

%[net,tr] = train(net,imagesBATCH,targetBATCH);
