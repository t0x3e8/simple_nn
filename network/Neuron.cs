
public class Neuron
{
    public double[] Weights { get; set; }
    public double Bias { get; set; }
    public double Output { get; private set; }
    public double Gradient { get; private set; }

    public Neuron(int inputCount)
    {
        this.Weights = new double[inputCount];
        this.Bias = 0.0;
        this.InitializeWeights();
    }

    public void CalculateGradient(double error)
    {
        double derivative = this.Output * (1 - Output);
        this.Gradient = error * derivative;
    }

    public void UpdateWeights(double [] inputs, double learnRate) {
        for (int i = 0; i < this.Weights.Length; i++)
        {
            this.Weights[i] -= learnRate * this.Gradient * inputs[i];
        }

        this.Bias -= learnRate * this.Gradient;
    }

    private void InitializeWeights()
    {
        Random rand = new Random();
        for (int i = 0; i < this.Weights.Length; i++)
        {
            this.Weights[i] = rand.NextDouble() * 2 - 1;
        }

        this.Bias = rand.NextDouble() * 2 - 1;
    }

    public double Activate(double[] inputs)
    {
        double activation = this.Bias;
        for (int i = 0; i < this.Weights.Length; i++)
        {
            activation += this.Weights[i] * inputs[i];
        }

        this.Output = this.Sigmoid(activation);
        return this.Output;
    }

    private double Sigmoid(double activation)
    {
        return 1.0 / (1 + Math.Exp(-activation));
    }
}