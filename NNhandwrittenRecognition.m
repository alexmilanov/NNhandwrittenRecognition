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
%Mean is equal to zero, standard deviation for bias values init is 1 and
%for weight initialization is equal to 1/sqrt(n) where n is number of
%weights TODO 530*784

IW = (1/sqrt(415520)) .* randn(S1,I);
b1 = randn(S1,1);

LW = (1/sqrt(5300)) .* randn(O, S1);
b2 = randn(O,1);

net.IW{1,1} = IW;
net.LW{2,1} = LW;
net.b{2} = b2;
net.b{1} = b1;

net.performFcn = 'msereg';
net.performParam.ratio = 0.5;
net.trainFcn = 'traingd';
net.trainParam.goal = 1e-3;
net.trainParam.show = 1;
net.trainparam.epochs = 300;
net.trainParam.lr = 0.2;
net.trainParam.max_fail = 75;
bn = 0;
trn = 0;
END = 500;
%net = init(net);
%VV.P,  TV.P  - Validation/test inputs.
%VV.T,  TV.T  - Validation/test targets, default = zeros.
%VV.Pi, TV.Pi - Validation/test initial input delay conditions, default = zeros.
%VV.Ai, TV.Ai - Validation/test layer delay conditions, default = zeros.
%validation input 20% of minibatch size
%input and target are 60%
%test input i.e. TV are 20% of minibatch

result = 0;
        add = 0;
        for x = 1:(END*0.4)
           if(add == 3)
               add = 2;
            elseif(add == 2 || add == 0)
               add = 3;
            end
            result = (result + add);
            index(x) = result;
        end

for bn = 1:END:60000
    trn = bn
    temp = trn;
        if(trn > 60000) break;
        end
        
        var = 1;
        VV.P = images(:, bn+2:5:bn + (END-1));
        bn = temp;
        TV.P = images(:, bn+4:5:bn + (END - 1));
        bn=temp;

        VV.T = target(:, trn + 2:5:trn + (END-1));
        trn = temp;
        TV.T = target(:, trn + 4:5:trn + (END-1));
        bn = temp;
        trn = temp;
        
        imagesBATCH = images(:, bn:bn + (END - 1));
        
        imagesBATCH(:, index) = [];
        
        targetBATCH = target(:, trn:trn + (END - 1));
        trn = temp;
        targetBATCH(:, index) = [];

        [net,tr, Y, E] = train(net,imagesBATCH,targetBATCH,[],[],VV,TV);
       
    
end
%--------------------------------------------------------------------
%Use of trained network

imagesUSE = loadMNISTImages('t10k-images.idx3-ubyte');
labelsUSE = loadMNISTLabels('t10k-labels.idx1-ubyte');

targetUSE = full(ind2vec(labelsUSE+1));
output = sim(net,imagesUSE);
true = 0;
false = 0;
for index = 1:1:10000
    [M,I] = max(output(:,index));
    I = I - 1;
    J = labelsUSE(index);
    if(I == J)
        true = true + 1;
    else
        false = false + 1;
    end
end
true = (true / 100)

[m,b,r]=postreg(output,targetUSE)