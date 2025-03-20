from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

import os 

os.environ["WANDB_DISABLED"] = "true"

import torch
import torch.distributed as dist

dist.init_process_group("nccl")  # Use NCCL backend for multi-GPU training


import json
with open('ev_assistant_bigger_dataset.json', 'r') as file:
    data_list = json.load(file)

import json
with open('ev_assistant_bigger_dataset_2.json', 'r') as file:
    data_list_2 = json.load(file)    

with open('intent_ev_dataset.json', 'r') as file:
    training_dataset = json.load(file) 
    
with open('testing_dataset.json', 'r') as file:
    testing_dataset = json.load(file) 
    
data_list=data_list+data_list_2

intents = ['battery_status_check', 'battery_health_tips', 'battery_range_estimate', 'battery_preconditioning', 'fast_charging_vs_slow_charging', 'charging_station_locator', 'optimal_charging_time', 'regenerative_braking_usage', 'battery_saving_tips', 'charging_safety_guidelines', 'cabin_preconditioning', 'climate_control_auto_mode', 'heated_seat_control', 'ventilated_seat_control', 'steering_wheel_heater_control', 'rear_defroster_usage', 'cabin_air_quality_monitoring', 'AC_vs_fan_usage', 'climate_control_zoning', 'window_tint_impact_on_temperature', 'sunroof_ventilation_control', 'humidity_control', 'cabin_temperature_adjustment', 'climate_control_scheduling', 'air_purifier_usage', 'cabin_noise_reduction', 'climate_control_remote_activation', 'smart_temperature_prediction', 'seat_memory_climate_preferences', 'bluetooth_pairing', 'wireless_carplay_setup', 'android_auto_configuration', 'voice_command_activation', 'GPS_navigation_usage', 'music_streaming_control', 'smartphone_integration', 'gesture_control_usage', 'in-car_wifi_setup', 'radio_tuning_commands', 'custom_audio_profiles', 'voice_assistant_commands', 'touchscreen_interface_guide', 'USB_device_connection', 'over_the_air_update_notification', 'app_store_functionality', 'cabin_speaker_adjustment', 'navigation_system_preferences', 'emergency_contact_setup', 'personalized_driver_profiles', 'lane_assist_activation', 'adaptive_cruise_control', 'blind_spot_warning', 'emergency_braking_activation', 'driver_monitoring_system', 'auto_headlight_adjustment', 'parking_assist_usage', 'collision_warning_system', 'child_lock_activation', 'remote_lock_status_check', 'sentry_mode_activation', 'auto_wiper_activation', 'emergency_call_function', 'rear_camera_activation', 'autonomous_parking', 'pedestrian_alert_system', 'driver_drowsiness_alert', 'passenger_airbag_control', 'speed_limit_advisor', 'remote_vehicle_shutdown', 'vehicle_location_tracking', 'geo_fencing_alerts', 'valet_mode_activation', 'brake_pad_wear_monitoring', 'energy_consumption_analytics', 'overheat_protection_system', 'sunroof_control', 'rear_climate_control', 'air_recirculation_mode', 'glovebox_control', 'steering_mode_adjustment', 'one_pedal_driving_mode', 'automatic_high_beam_control', 'driving_mode_selection', 'lane_departure_warning', 'traffic_sign_recognition', 'hill_assist_control', 'side_mirror_auto_fold', 'smart_routing_for_ev_trips', 'wireless_phone_charging', 'keyless_entry_system', 'voice_control_for_windows', 'rear_seat_reminder_alert', 'driver_profile_memory', 'heated_mirror_activation', 'ambient_lighting_customization', 'adaptive_suspension_control', 'pedal_response_adjustment', 'eco_mode_activation', 'passenger_display_control', 'auto_trunk_opening', 'smart_summon_feature', 'rear_collision_warning', 'power_frunk_control', 'speed_limit_mode', 'child_seat_detection', 'smart_rearview_mirror', 'remote_climate_control', 'charging_port_locking', 'battery_heater_activation', 'navigation_energy_prediction', 'winter_driving_mode', 'phone_as_key_functionality', 'digital_assistant_integration', 'driver_attention_monitor', 'emergency_steering_assist', 'tire_pressure_monitoring', 'cabin_air_filter_status', 'regen_braking_intensity_control', 'passenger_air_vent_control', 'key_fob_battery_status', 'automatic_seat_adjustment', 'safety_exit_warning', 'rear_cross_traffic_alert', 'pet_mode_activation', 'trailer_mode_activation']
entities = [
    "battery_level", "charging_speed", "charging_station_type", "charger_connector_type", "battery_health_status",
    "charging_time_remaining", "charging_cost", "charging_location", "preconditioning_status", "home_charging_status",
    "DC_fast_charger_availability", "battery_temperature", "state_of_charge", "battery_swap_availability",
    "grid_load_during_charging", "solar_panel_integration", "charging_subscription_status", "energy_tariff_plan",
    "charger_fault_code", "bidirectional_charging_status", "speed", "current_gear", "acceleration_pattern",
    "tire_pressure", "regen_braking_level", "driving_mode", "motor_efficiency", "torque_distribution",
    "range_prediction", "hill_ascent_status", "wind_resistance", "drag_coefficient", "battery_consumption_rate",
    "average_energy_consumption", "road_gradient", "tire_wear_status", "real_time_range", "current_energy_usage",
    "wheel_alignment_status", "one_pedal_driving_status", "current_temperature", "HVAC_mode", "heated_seat_status",
    "ventilated_seat_status", "steering_wheel_heating", "humidity_level", "air_quality_index", "cabin_temperature_preference",
    "AC_fan_speed", "rear_defroster_status", "cabin_air_filter_status", "air_purifier_mode", "climate_control_zones",
    "seat_position_profile", "driver_preference_profile", "climate_schedule", "window_tint_level", "sunroof_position",
    "noise_cancellation_mode", "smart_temperature_adjustment", "connected_device_name", "bluetooth_status",
    "WiFi_signal_strength", "audio_volume_level", "current_radio_station", "active_media_source", "carplay_status",
    "android_auto_status", "navigation_destination", "GPS_signal_quality", "voice_command_recognition",
    "in-car_app_status", "over_the_air_update_version", "personalized_audio_profile", "touchscreen_sensitivity",
    "gesture_control_mode", "speaker_balance", "screen_brightness_level", "navigation_preference", "media_playback_status",
    "lane_assist_status", "adaptive_cruise_control_status", "blind_spot_warning_status", "collision_warning_intensity",
    "parking_assist_status", "sentry_mode_status", "auto_braking_status", "airbag_status", "child_lock_status",
    "driver_alertness_level", "emergency_brake_activation", "remote_lock_status", "speed_limit_alert",
    "vehicle_location", "auto_wiper_sensitivity", "proximity_alert_status", "autonomous_parking_mode",
    "pedestrian_alert_sound_level", "remote_start_status", "passenger_detection_status"
]

system_prompt = f'''You are an EV assistant providing concise and accurate responses to user queries. 
Your responses must follow this structured JSON format:

{{
  "response": "Concise response (max 15 words)",
  "intent": "Relevant intent from the provided list",
  "entities": ["Relevant entities from the provided list"]
}}

Guidelines:
1. **Response Length**: Your response must be **clear and limited to 15 words or fewer**.
2. **Intent Selection**: Choose the **most relevant** intent from the predefined list: {", ".join(intents)}
3. **Entity Identification**: Extract and include the **correct entities** from the predefined list: {", ".join(entities)}
4. **Context Awareness**: Ensure your responses match the user's query **without unnecessary details**.
5. **No Extra Text**: Only return the JSON output, with no explanations or additional comments.

### Example Inputs & Outputs:

**User Input:** "How much range do I have left?"
**Expected Output:**
```json
{{
  "response": "You have approximately 220 miles remaining.",
  "intent": "battery_range_estimate",
  "entities": ["range_prediction", "real_time_range"]
}}
'''

def tokenization_qwen_model(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text

import pandas as pd 
complte_multiturn_conversation=[]
for data in data_list:
    try: 
        for key, value in data[0].items():
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            for i in range(0, len(value[0]['conversation']), 2):
                question=data[0][key][0]['conversation'][i]["content"]
                entities=data[0][key][0]['conversation'][i+1]["entities"]
                answer=data[0][key][0]['conversation'][i+1]["content"]
                
                messages.append({"role": "user", "content": question})
                response={"response": answer, "intent":key, "entities": entities }
                messages.append({"role": "assistant", "content": str(response)})
            
            tokenized_text=tokenization_qwen_model(messages) 
            complte_multiturn_conversation.append(tokenized_text)
    except:
        continue

print("complete")
training_complte_multiturn_conversation=[]
for key, value in training_dataset.items():
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        for i in range(0, len(value[0]['conversation']), 2):
            question=value[0]['conversation'][i]["content"]
            entities=value[0]['conversation'][i]["entities"]
            answer=value[0]['conversation'][i+1]["content"]
            
            messages.append({"role": "user", "content": question})
            response={"response": answer, "intent":key, "entities": entities }
            messages.append({"role": "assistant", "content": str(response)})   
            
        tokenized_text=tokenization_qwen_model(messages) 
        training_complte_multiturn_conversation.append(tokenized_text) 
    

print("complete")

testing_complte_multiturn_conversation=[]
for key, value in testing_dataset.items():
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        for i in range(0, len(value[0]['conversation']), 2):
            question=value[0]['conversation'][i]["content"]
            entities=value[0]['conversation'][i]["entities"]
            answer=value[0]['conversation'][i+1]["content"]
            
            messages.append({"role": "user", "content": question})
            response={"response": answer, "intent":key, "entities": entities }
            messages.append({"role": "assistant", "content": str(response)})
            
        tokenized_text=tokenization_qwen_model(messages) 
        testing_complte_multiturn_conversation.append(tokenized_text) 

complte_multiturn_conversation=complte_multiturn_conversation+training_complte_multiturn_conversation

train_data=pd.DataFrame()
train_data['text']=complte_multiturn_conversation

test_data=pd.DataFrame()
test_data['text']=testing_complte_multiturn_conversation

import os
import sys
import logging
import pandas as pd
import torch
import transformers
import torch.distributed as dist
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

###################
# Hyperparameters
###################
training_config = {
    "bf16": True,
    "do_eval": True,
    "learning_rate": 2e-5,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 5,
    "max_steps": -1,
    "output_dir": "./intelli_EV_assistant_adapter_v2",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "fp16": False,  # Ensure mixed precision is disabled
    "bf16_full_eval": True,
}

#############################
# QLoRA (4-bit Quantization)
#############################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}

train_conf = SFTConfig(
    **training_config,
    ddp_find_unused_parameters=False,  # Optimize multi-GPU training
    dataloader_num_workers=4,  # Adjust based on system specs
    torch_compile=False,  # Ensure stability
    max_seq_length=4096,  # Updated to 12,288
    dataset_text_field="text",
    packing=True,
)

peft_conf=LoraConfig(**peft_config)

###############
# Setup Logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)

logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")


################
# Model Loading
################
checkpoint_path = "microsoft/Phi-3.5-mini-instruct"  # Phi-4 Model
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={'':torch.cuda.current_device()},
    quantization_config=bnb_config
)

model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

# Setting sequence length to 12,288
tokenizer.model_max_length = 5120
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Convert to Hugging Face Dataset
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_data)
test_dataset=Dataset.from_pandas(test_data)

print("===================train_dataset====================")
print(train_dataset)
print("=======================================")

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,

)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = "left"
metrics = trainer.evaluate()
metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(train_conf.output_dir)

# Shutdown process group
dist.destroy_process_group()



