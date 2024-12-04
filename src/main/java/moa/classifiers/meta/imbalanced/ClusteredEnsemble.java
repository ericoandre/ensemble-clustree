package moa.classifiers.meta.imbalanced;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.clustree.ClusTree;

import moa.core.Measurement;

import moa.options.ClassOption;

/*
 * <p> Parameters:</p> <ul>
 * <li>-l : modelo base de Classificador. default HoeffdingTree</li>
 * </ul>
 */
public class ClusteredEnsemble extends AbstractClassifier  implements MultiClassClassifier {
    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 
    		'l',  "Classifier to train.", Classifier.class, "trees.HoeffdingTree -g 50 -c 0.01");

    // Inicializar o Clustree
    private ClusTree clusTree = new ClusTree();
    // modelos base vinculados a subespacos pelos hashcode
    private Map<Integer, Classifier> clusterBaseModels = new HashMap<>();

    boolean Start = false;

    // ------------------------------------------------------------------------------- //

    // retorna o hashcode do cluster com maior probabilidade de conter a instancia
    private Integer findCluster(Instance inst){
        // Obter microclusters do ClusTree
        Clustering clustersResult = this.clusTree.copy().getMicroClusteringResult();
        if(clustersResult!=null && !Start){
            for(int i =0; i< clustersResult.size(); i++){
                Cluster e = clustersResult.get(i);
                double exitInMcT = e.getInclusionProbability(inst);
                if(exitInMcT>0)
                    return e.hashCode();
            }
        }
        return null;
    }
    /**
     * Atualiza os microclusters no ClusTree e gerencia os modelos base associados a cada cluster.
     * 
     * @param inst Instância a ser utilizada para atualização e treinamento.
     */ 
    private void updateClust(Instance inst) {
        // Mapa para rastrear os microclusters existentes
        Map<Integer, Cluster> microclustersRemov = new HashMap<>();
        
        // Obter microclusters do ClusTree
        Clustering clustersResult = this.clusTree.copy().getMicroClusteringResult();
        if (clustersResult == null || clustersResult.size() == 0) {
            return; // Sem clusters para processar
        }
        
        // Processa os microclusters
        for (int i = 0; i < clustersResult.size(); i++) {
            Cluster e = clustersResult.get(i);
            if (e == null) continue;
            double inclusionProbability = e.getInclusionProbability(inst);
            microclustersRemov.put(e.hashCode(), e); // Adiciona ao mapa de microclusters
            if (inclusionProbability > 0) {
                Integer clusterId = e.hashCode();
                // Treina o modelo base para o cluster
                Classifier model = CreateBaseModels(clusterId);
                // Se a instância esta rotulada treina o modelo base
                if (!inst.classIsMissing()){
                    if (model != null) model.trainOnInstance(inst);
                }
            }
        }
        // Remove os modelos base para clusters inexistentes
        List<Map.Entry<Integer, Classifier>> entries = new ArrayList<>(clusterBaseModels.entrySet());
        for(int i =0; i< clusterBaseModels.size(); i++){
            Integer key = entries.get(i).getKey();
            if(!microclustersRemov.containsKey(key)){
                RemoveBaseModels(key);
            }
        }
    }
    
    // ------------------------------------------------------------------------------- //
    
    /**
     * Obtém o modelo base associado a um cluster específico.
     *
     * @param clusterId O identificador único do cluster cujo modelo base deve ser retornado.
     * @return O modelo base associado ao cluster.
     */
    private Classifier getBaseModel(Integer clusterId) {
        if (this.clusterBaseModels != null){
            Classifier model = this.clusterBaseModels.get(clusterId);
            if (model != null){
                return model;
            }
        }
        return null;
    }
    /**
     * Cria um modelo base associado ao cluster identificado pelo clusterId.
     * 
     * @param clusterId Identificador único do cluster.
     * @return O modelo base associado ao cluster ou um novo modelo se ele não existir.
     */
    private Classifier CreateBaseModels(Integer clusterId) {
        Classifier baseClassifier = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        // HoeffdingTree baseClassifier = new HoeffdingTree();
        baseClassifier.prepareForUse(); // Configura as opções padrão
        baseClassifier.resetLearning(); // Reinicia o aprendizado
        return clusterBaseModels.computeIfAbsent(clusterId, id -> baseClassifier);
    }
    /**
     * Remove o modelo base associado a um cluster.
     * 
     * @param clusterId Identificador único do cluster cujo modelo será removido.
     */
    private void RemoveBaseModels(Integer clusterId) {
        if(this.clusterBaseModels.containsKey(clusterId)){
            this.clusterBaseModels.remove(clusterId);
        }
    }
    
    // ------------------------------------------------------------------------------- //

    // Inicializar o conjunto de modelos e o ClusTree
    private void initEnsemble(){
        // Inicializar o Clustree
        this.clusTree = new ClusTree();
        this.clusTree.prepareForUse();
        this.Start = true;
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if(this.clusTree == null) this.initEnsemble();

        // Atualiza o ClusTree com a nova instância
        this.clusTree.trainOnInstanceImpl(inst.copy());
        this.updateClust(inst);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        Integer clusterId = this.findCluster(inst.copy());
        Classifier model = this.getBaseModel(clusterId);
        return model != null ? model.getVotesForInstance(inst) : new double[0];
    }

    @Override
    public void resetLearningImpl() {}

    @Override
    public boolean isRandomizable() {return false;}

    @Override
    protected Measurement[] getModelMeasurementsImpl() {return new Measurement[0];}

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        String indentation = " ".repeat(indent);
        out.append(indentation).append("Modelo baseado em clusters utilizando ClusTree.\n");
        out.append(indentation).append("Características principais:\n");
        out.append(indentation).append(" - Atualização contínua com instâncias do fluxo.\n");
        out.append(indentation).append(" - Treinamento com a instância mais recente de cada cluster.\n");
        out.append(indentation).append(" - Subespaços definidos por ClusTree.\n");
    }
}