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
}