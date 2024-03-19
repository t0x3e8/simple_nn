// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, Neural Network!");

NeuralNetwork nn = new NeuralNetwork(3, 2, 1);

double[] inputs = { 0.5, 0.1, 0.4 };
double[] outputs = nn.FeedForward(inputs);

Console.WriteLine($"Output: {outputs[0]}");

// https://chat.openai.com/share/ba912eca-6b4e-42d7-b36f-958deb3a3ab1