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
}