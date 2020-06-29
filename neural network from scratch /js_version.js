// pure js neural network

/////////////////////////////
//     author: grafstor
//     date: 29.06.20
/////////////////////////////

// verison 1.0 

function dot(a, b) {
	let result = [];
	for (let i = 0; i<a.length; i++){
		for (let j = 0; j<b[0].length; j++){
			let summary = 0;
			for (let p = 0; p<a[i].length; p++)
				summary = summary + a[i][p]*b[p][j];

			result.push(summary);
		}
	}
	return result;
}
function sigmoid(x) {
	let result = [];
	for (let i = 0; i<x.length; i++)
		result.push(1 / (1 + Math.exp(-x[i])));
	return result;
}
function sigmoid_derivative(x) {
	let result = [];
	for (let i = 0; i<x.length; i++)
		result.push(x[i]*(1-x[i]));
	return result;
}
function count_error(y,p) {
	let result = [];
	for (let i = 0; i<y.length; i++)
		result.push(y[i] - p[i]);
	return result;
}
function count_delta(e,d) {
	let result = [];
	for (let i = 0; i<e.length; i++)
		result.push([e[i]*d[i]]);
	return result;
}
function T(a) {
	let result = [];
	for (let i = 0; i<a[0].length; i++){
		let second = [];
		for (let j = 0; j<a.length; j++)
			second.push(a[j][i]);
		result.push(second);
	}
	return result;
}

let weights = [[0.23],[-0.46],[0.53]];
let train_x = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]];
let train_y = [0, 1, 1, 0];

for (let epoch = 0; epoch<10000; epoch++){
	output = sigmoid(dot(train_x, weights));
	error = count_error(train_y, output);
	delta = count_delta(error, sigmoid_derivative(output));
	adjustment = dot(T(train_x), delta);

	for (var i = 0; i < weights.length; i++)
			weights[i][0] += adjustment[i];
}

console.log(sigmoid(dot([[1,0,1]], weights)));