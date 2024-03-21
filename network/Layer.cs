public class Layer {
    public Neuron [] Neurons { get; set; }
    
    public Layer(int neuronCount, int inputCount)
    {
        this.Neurons = new Neuron[neuronCount];
        for( int i =0; i< neuronCount; i++) {
            this.Neurons[i] = new Neuron(inputCount);
        }
    }

    public double [] Activate(double [] inputs) {
        return Neurons.Select(n => n.Activate(inputs)).ToArray();
    }

    public void Backpropagate(double [] nextLayerGradients, double [][] nextLayerWeight) {
        for (int i = 0; i < this.Neurons.Length; i++)
        {
            double error = 0;
            for (int j = 0; j < nextLayerGradients.Length; j++)
            {
                error += nextLayerGradients[j] * nextLayerWeight[j][i];
            }
            this.Neurons[i].CalculateGradient(error);
        }
    }

    public void UpdateWeights(double[] inputs, double learnRate) {
        foreach (var neuron in this.Neurons)
        {
            neuron.UpdateWeights(inputs, learnRate);
        }
    }
}