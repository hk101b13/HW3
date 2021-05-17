#include "mbed.h"
#include "mbed_rpc.h"
#include "math.h"
#include "uLCD_4DGL.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "stm32l475e_iot01_accelero.h"

#define PI 3.141592965
Config config;

//-----------------------------------------------------------------------------------------------------------
// MQTT
WiFiInterface *wifi;
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;
const char* topic = "Mbed";
Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;

BufferedSerial pc(USBTX, USBRX);
DigitalOut led1(LED1);
InterruptIn sw1(USER_BUTTON);
EventQueue queue(32 * EVENTS_EVENT_SIZE);

uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

Thread t;
int threshold = 30;
double angel;
double length_v;
double length_g;
double sum;
// Whether we should clear the buffer next time we fetch data
bool should_clear_buffer = false;
bool got_data = false;

// The gesture index of the prediction
int gesture_index;

Ticker flipper;

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
MQTT::Client<MQTTNetwork, Countdown> * client_out;
int WIFI_FUNCTION();

int machine_learning();
void tilt_ang_det(int threshold);
RPCFunction rpcGesture(&machine_learning, "gesture_UI");
RPCFunction rpcTilt(&tilt_ang_det, "tilt_angel");


void publish_message();


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
int WIFI_FUNCTION(){
   
    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
            printf("ERROR: No WiFiInterface found.\r\n");
            return -1;
    }


    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            return -1;
    }


    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

    //TODO: revise host to your IP
    const char* host = "192.168.105.51";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
            printf("Connection error.");
            return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }

    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    // flipper.attach(mqtt_queue.event(&publish_message, &client), 1500ms);

    int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }

    while (1) {
            if (closed) break;
            client.yield(500);
            ThisThread::sleep_for(500ms);
    }

    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");
    client_out  =  &client;
    return 0;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------
void publish_message(){
    led1=1;
    message_num++;
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "%d", threshold);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client_out->publish(topic, message);

    printf("rc:  %d\r\n", rc);
    printf( buff);
    led1=0;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
void close_mqtt() {
    closed = true;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
int count_theda(){
    int16_t pDataXYZ[3] = {0};
    int16_t g[3] = {0, 0 ,100};
    int16_t s[3] = {0};
    BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    for (int i=0; i<3; i++){
        s[i] =pDataXYZ[i]*g[i];
    }
    for(int i = 0; i<3; i++){
            sum+=s[i];
    }
    length_v = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    length_g = sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]);
    angel = 90.0 - acos(sum/(length_v*length_g))*180.0/PI;

    return angel;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------
// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}



//--------------------------------------------------------------------------------------------------------------------------------------------------------
int machine_learning(){

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");
  while (true) {

    if(sw1 == 1){
      uLCD.printf("%d",threshold);
      return 0;
    } 
    else{
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }
      
    if(gesture_index == 1){
        threshold=threshold+5;
        if(threshold==50){
            threshold = 30;
        }
        //printf("%d\n",threshold);
    }
    else if(gesture_index == 2){
        threshold=threshold-5;
        if(threshold==30){
            threshold = 50;
        }
        //printf("%d\n",threshold);
    }
        return threshold;
    }
  }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------------------------------
void tilt_ang_det(int threshold){
    count_theda();
    if (angel>threshold){
    for(int i=0;i<10;i++){
      publish_message();
      }
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

int main() {
  for(int i=0;i<10;i++){
    printf("start1");
  }
   config.seq_length=64;

   config.consecutiveInferenceThresholds[0]=25;
   config.consecutiveInferenceThresholds[1]=10;
   config.consecutiveInferenceThresholds[2]=20;

   config.output_message[0]="right";
   config.output_message[1]="left";
   config.output_message[0]="left";
  //  t.start(WIFI_FUNCTION);

   BSP_ACCELERO_Init();

    char buf[256], outbuf[256];

    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");

  sw1.rise(&publish_message);
  printf("start");
    while(1) {
        memset(buf, 0, 256);
        for (int i = 0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }

        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }
}