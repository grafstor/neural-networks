(* pure pascal neural network*)

(*/////////////////////////*)
(*    author: grafstor*)
(*    date: 27.06.20*)
(*/////////////////////////*)

(*verison 2.0*)

program NN;

var 
    train_x: array [0..11] of integer = (0,0,1,  1,1,1,  1,0,1,  0,1,1);
    train_y: array [0..3] of real = (0,1,1,0);
    weigths: array [0..2] of real = (-0.145, -0.993, -0.359);
    
    dot_product: array [1..4] of real;
    output: array [1..4] of real;
    
    error: array [1..4] of real;
    sigmoid_derivative: array [1..4] of real;
    delta: array [1..4] of real;
    adjustment: array [1..3] of real;
    train_x_reversed: array [0..11] of integer = (0,1,1,0, 0,1,0,1, 1,1,1,1);
    
    test: array [0..2] of real = (1,0,1);
    
    summary, result: real;
    
    i, p, epoch: integer;
    
function sigmoid(x:real): real;
begin
    sigmoid := 1 / (1 + exp(-x*LN(2.71828182846)));
end;

function dsigmoid(x:real): real;
begin
    dsigmoid := x*(1 - x);
end;

begin
    
    for  epoch := 0 to 1200 do
    begin
        
        for  i := 0 to 3 do
        begin
            summary := 0;
            for p := 0 to 2 do
                summary := summary + train_x[i*3+p] * weigths[p];
            dot_product[i] := summary;
        end;
        
        for i := 0 to 3 do
            output[i] := sigmoid(dot_product[i]);

        for i := 0 to 3 do
            error[i] := train_y[i] - output[i];
    
        for i := 0 to 3 do
            sigmoid_derivative[i] := dsigmoid(output[i]);
        
        for i := 0 to 3 do
            delta[i] := error[i] * sigmoid_derivative[i];

        for  i := 0 to 2 do
        begin
            summary := 0;
            for p := 0 to 3 do
                summary := summary + train_x_reversed[i*4+p] * delta[p];
            adjustment[i] := summary;
        end;
        
        for i := 0 to 2 do
            weigths[i] := weigths[i] + adjustment[i];
    end;
    
    summary := 0;
    for p := 0 to 2 do
        summary := summary + test[p] * weigths[p];
    result := sigmoid(summary);

    write('test result: ');
    writeln(result);
    
    writeln('weigths: ');
    for i := 0 to 2 do
        writeln(weigths[i]);
end.