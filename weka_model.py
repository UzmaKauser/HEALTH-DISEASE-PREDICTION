import weka.core.jvm as jvm
import weka.core.converters as converters
import weka.classifiers as classifiers

def load_weka_model(model_path):
    jvm.start()
    
    classifier = classifiers.Classifier(jobject=classifiers.Classifier(jobject=None))
    classifier.load(model_path)
    
    return classifier
