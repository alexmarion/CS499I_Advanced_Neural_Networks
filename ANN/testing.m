% Testing stuff
A = [0,0,0,0;   % Training data
    1,1,1,1;
    2,2,2,2;
    3,3,3,3];
B = [1;1;2;2];  % Training classes

C = [];         % Testing data
D = [];         % Testing classes

% A(B == 1,:); % Find all rows where class == 1


% random number from 1 to number of images in class
a = 1;
b = size(A(B == 1, :),1);
r = randi([a b],1,1);   % Single random integer

% Add to testing data
C(end+1,:) = A(r,:);
D(end+1,:) = B(r,:);

% Remove from training data
A(r,:) = [];
B(r,:) = [];



