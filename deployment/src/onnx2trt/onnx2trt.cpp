#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <logger.h>


using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;


void _serialize_engine(ICudaEngine* engine)
{
  IHostMemory* modelStream{ nullptr }; 
  modelStream = engine->serialize(); 
  ofstream f("model.engine", ios::binary); 
  // f << modelStream->data();
  f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
  f.close();
}


int main(int argc, char** argv) {
  // Create builder 
  Logger logger;
  IBuilder* builder = createInferBuilder(logger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilderConfig* config = builder->createBuilderConfig(); 
  
  // Create model to populate the network 
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

  // Parse ONNX file 
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

  // Build engine
  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(1 << 30);  // 1GB
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  _serialize_engine(engine);
  
  engine->destroy();
  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  return 0;
}
