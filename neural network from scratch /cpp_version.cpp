// pure c++ neural network

///////////////////////////
//    author: grafstor
//    date: 25.06.20
///////////////////////////

// verison 1.0

#include <iostream>
#include <vector>
#include <math.h>

void print_2d_vector(std::vector<std::vector<float>> array)
{
	for (int i = 0; i<array.size(); i++)
	{
		for (int j = 0; j<array[i].size(); j++)
			std::cout << array[i][j];
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void print_1d_vector(std::vector<float> array)
{
	for (int i = 0; i<array.size(); i++)
	{
		std::cout << array[i];
		std::cout << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
}


void dot(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b, std::vector<std::vector<float>> &dot_product)
{
	for (int i = 0; i<a.size(); i++)
	{		
		std::vector<float> second;

		for (int j = 0; j<b[0].size(); j++)
		{
			float summary = 0.0;
			for (int p = 0; p<a[i].size(); p++)
			{
				summary += a[i][p]*b[p][j];
			}
			second.push_back(summary);
		}
		dot_product.push_back(second);
	}
}

void sigmoid(std::vector<std::vector<float>> a, std::vector<float> &result)
{
	const float E = 2.71828182846;

	for (int i = 0; i<a.size(); i++)
	{
		for (int j = 0; j<a[i].size(); j++)
		{
			float x =  1 / (1 + pow(E, -a[i][j]));

			result.push_back(x);
		}
	}
}

void sigmoid_derivative(std::vector<float> a, std::vector<float> &result)
{

	for (int i = 0; i<a.size(); i++)
	{
		float x =  a[i] * (1 - a[i]);

		result.push_back(x);
	}
}

void T(std::vector<std::vector<float>> a, std::vector<std::vector<float>> &result)
{

	for (int i = 0; i<a[0].size(); i++)
	{
		std::vector<float> tmp;
		for (int j = 0; j<a.size(); j++)
		{
			float x =  a[j][i];

			tmp.push_back(x);
		}
		result.push_back(tmp);
	}
}

int main()
{

	std::vector<std::vector<float>> weights{{0.23},{-0.46},{0.53}};
	std::vector<std::vector<float>> train_x{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};
	std::vector<float> train_y{0, 1, 1, 0};


	for (int epoch = 0; epoch<1500; epoch++)
	{
		// train vectors
		std::vector<std::vector<float>> dot_product;
		std::vector<float> output;
		std::vector<float> error;
		std::vector<float> dsigmoid_result;
		std::vector<std::vector<float>> delta;
		std::vector<std::vector<float>> T_train_x;
		std::vector<std::vector<float>> adjustment;

		// feed forword
		dot(train_x, weights, dot_product);
		sigmoid(dot_product, output);

		// count error
		for (int i = 0; i<output.size(); i++)
			error.push_back(train_y[i] - output[i]);

		// count sigmoid derivative
		sigmoid_derivative(output, dsigmoid_result);

		// count sigmoid delta
		for (int i = 0; i<error.size(); i++)
		{
			std::vector<float> tmp;
			tmp.push_back(dsigmoid_result[i]*error[i]);
			delta.push_back(tmp);
		}

		// count adjustment
		T(train_x, T_train_x);
		dot(T_train_x, delta, adjustment);

		// change weights
		for (int i = 0; i<weights.size(); i++)
			weights[i][0] += adjustment[i][0];
	}
	// test vectors
	std::vector<std::vector<float>> test{{1, 0, 1}};
	std::vector<std::vector<float>> dot_product;
	std::vector<float> output;

	// feed forword
	dot(test, weights, dot_product);
	sigmoid(dot_product, output);

	// print test result
	std::cout << output[0] << "result" <<std::endl;

	// print weights
	std::cout << "weights: " <<std::endl;
	print_2d_vector(weights);

	return 0;
}
