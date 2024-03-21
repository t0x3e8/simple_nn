using System.Runtime.InteropServices;

public class NeuralNetwork
{
    public Layer InputLayer { get; set; }
    public Layer HiddenLayer { get; set; }
    public Layer OutputLayer { get; set; }

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        this.InputLayer = new Layer(inputSize, inputSize);
        this.HiddenLayer = new Layer(hiddenSize, inputSize);
        this.OutputLayer = new Layer(outputSize, hiddenSize);
    }

    public double[] FeedForward(double[] inputs)
    {
        var hiddenOutput = this.HiddenLayer.Activate(inputs);
        var finalOutput = this.OutputLayer.Activate(hiddenOutput);

        return finalOutput;
    }

    public void Train(double [] inputs, double[] expected, double learnRate) {
        double [] outputs = this.FeedForward(inputs);

        // Calculate output layer error
        double[] outputErrors = new double[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
        {
            outputErrors[i] = expected[i] - outputs[i];
        }

        // Calculate gradients for output layer
        for (int i = 0; i < this.OutputLayer.Neurons.Length; i++)
        {
            this.OutputLayer.Neurons[i].CalculateGradient(outputErrors[i]);
        }

        // Backpropagate from output layer to hidden layer
        this.HiddenLayer.Backpropagate(
            this.OutputLayer.Neurons.Select(n => n.Gradient).ToArray(),
            this.OutputLayer.Neurons.Select(n => n.Weights).ToArray());

        // Update Weights and biases
        this.OutputLayer.UpdateWeights(this.HiddenLayer.Neurons.Select(n => n.Output).ToArray(), learnRate);
        this.HiddenLayer.UpdateWeights(inputs, learnRate);
    }
}