
internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, Neural Network!");

        // Default - not trained - network
        NeuralNetwork nn = new(2, 2, 1);        
        double[] inputs = [1, 1];
        double[] outputs = nn.FeedForward(inputs);
        Console.WriteLine($"For Input [{inputs[0]}, {inputs[1]}], the default Output is {outputs[0]}");
    }
}