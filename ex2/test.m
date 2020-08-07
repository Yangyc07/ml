X = [1 3; 2 3; 3 5; 4 7; 5 8; 6 10];
y = [1; 0; 0; 1; 0; 1];
for i = 1:size(X, 1),
      plot(X(i, 1), X(i, 2), 'ko', 'MarkerFaceColor', 'y');
      plot(X(2, 1), X(2, 2), 'ko', 'MarkerFaceColor', 'y');
end