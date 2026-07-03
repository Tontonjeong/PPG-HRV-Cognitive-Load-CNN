/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32f4xx.h"
#include "arm_math.h"
#include "stm32f4xx_hal.h"
#include "arm_const_structs.h"
#include "stdio.h"
#include "string.h"
#include "math.h"   // fabsf

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define numStage_PPG 2            // HP(0.5Hz) 1차 + LP(8Hz) 1차 = 총 2 스테이지
#define blocksize    1
#define numReadings  5

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint32_t ADCValue[1]; //ADC의 값, ADC[1]에 넣음으로 과거값을 넣는다.
char RcvData[256]; //UART를 위한 버퍼
static uint8_t just_peaked = 0;

uint32_t window[numReadings] = {0};
uint32_t readIndex = 0;
uint32_t total = 0;
uint32_t average = 0;

uint32_t Smooth_PPG = 0;
float32_t Raw_Data = 0;
float32_t IIR_Output_PPG = 0;

arm_biquad_cascade_df2T_instance_f32 S1;            // IIR 필터 핸들
static float32_t pState_PPG[numStage_PPG * 2] = {0};

// 전압 변환 (pow 제거)
static inline float32_t ADC_Float_DATA(uint32_t adc)
{
    return ((float32_t)adc * 3.3f / 4095.0f);
}

// <Coefficient>  (각 스테이지 {b0,b1,b2,a1,a2} 순서, CMSIS-DF2T 규약에 맞춘)
// [0..9]  : 0.5Hz HPF(2차) × 1 stage
// [10..19]: 8Hz   LPF(2차) × 1 stage
static float32_t IIR_Low_Coeffs_PPG[numStage_PPG * 5] = {
	0.9937550f, -1.9875101f, 0.9937550f,   1.9874523f, -0.9875679f,
	0.0674553f,  0.1349106f, 0.0674553f,   1.1429805f, -0.4128016f,
};

// 이벤트/IBI 관련 파라미터
#define MAX_IBI            100
const uint32_t IBI_MIN_MS     = 250;
const uint32_t IBI_MAX_MS     = 2000;
const uint32_t REFRACTORY_US  = 300000U; // 300 ms (TIM2 1MHz 기준)

// 적응형 임계(엔벌로프)
float32_t env = 0.0f;
const float a_up = 0.40f;
const float a_dn = 0.02f;
const float TH_K = 0.15f;


// 상태 변수 (IBI)
uint32_t lastPeak_us = 0;
uint32_t ibi_ms[MAX_IBI] = {0};
uint16_t ibiIndex = 0;
volatile uint32_t currentIBI_ms = 0;
volatile uint8_t  ibiValid = 0;
uint8_t           ibiUpdated = 0;

// 미분 관련 상태
float32_t prevPPG = 0.0f;
float32_t prevDer1 = 0.0f;
float32_t prevDer2 = 0.0f;

// 이벤트 타임스탬프(디버깅/표시용)
uint32_t t_upstart_us = 0;
uint32_t t_slopemax_us = 0;
uint32_t t_peak_us = 0;

typedef enum { ST_IDLE, ST_RISING, ST_SLOPEMAXED } BeatState;
static BeatState st = ST_IDLE;

uint8_t upstart_hit = 0;
uint8_t slopemax_hit = 0;
uint8_t peak_hit = 0;
static uint32_t last_valid_ibi_ms = 0;   // PEAK에서 유효 IBI 확정 시 갱신

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM2_Init(void);

/* USER CODE BEGIN PFP */
// (불필요한 'A' 문자 제거, 필요 시 다른 프로토타입 추가)
static inline uint32_t t_us(void);
static inline uint32_t dt_us(uint32_t now, uint32_t then);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_TIM1_Init();
  MX_USART2_UART_Init();
  MX_TIM2_Init();
  HAL_TIM_Base_Start(&htim2);
  /* USER CODE BEGIN 2 */
  // 1. IIR 필터 초기화
  arm_biquad_cascade_df2T_init_f32(&S1, numStage_PPG, IIR_Low_Coeffs_PPG, pState_PPG);
  // 2. 타이머 인터럽트 (현재는 폴링 샘플링이지만 기존대로 유지) // <<< ADDED (원래 코드 유지)
  HAL_TIM_Base_Start_IT(&htim1);
  HAL_TIM_Base_Start(&htim2);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  // --- ADC 샘플 수집 (기존 코드 유지) ---
	      HAL_ADC_Start(&hadc1);
	      HAL_ADC_PollForConversion(&hadc1, 10);
	      ADCValue[0] = HAL_ADC_GetValue(&hadc1);

	      // 이동평균 스무딩 (기존)
	      total -= window[readIndex];
	      window[readIndex] = ADCValue[0];
	      total += window[readIndex];
	      readIndex++;
	      if (readIndex >= numReadings) readIndex = 0;
	      Smooth_PPG = total / numReadings;

	      // 전압 변환
	      Raw_Data = ADC_Float_DATA(ADCValue[0]);

	      // IIR 필터
	      arm_biquad_cascade_df2T_f32(&S1, &Raw_Data, &IIR_Output_PPG, blocksize);

	      // ===== 이벤트 검출 로직 (HRV 제거된 버전) =====

	      // 1) 엔벌로프(동적 임계)
	      float32_t xabs = fabsf(IIR_Output_PPG);
	      env = (xabs > env) ? (a_up * xabs + (1.0f - a_up) * env)
	                         : (a_dn * xabs + (1.0f - a_dn) * env);
	      float32_t thr = TH_K * env;

	      // 2) 1차/2차 미분
	      float32_t der1 = IIR_Output_PPG - prevPPG;
	      float32_t der2 = der1 - prevDer1;

	      // 3) timestamp (µs)
	      uint32_t now_us = t_us();

	      // ===== FSM =====
	      // IDLE: 임계 하에서 하강의 끝(상승 시작점)이 나타나면 UPSTART로 진입
	      if (st == ST_IDLE) {
	          // UPSTART: 데이터<thr, 현재<이전, 1차: 음→양
	    	  if ((IIR_Output_PPG < thr) &&
	    	      (IIR_Output_PPG >= prevPPG) &&          // ★ 이전값보다 커지기 시작(상승)
	    	      (prevDer1 < 0.0f && der1 >= 0.0f))   // ★ 음→양
	          {
	              t_upstart_us = now_us;
	              upstart_hit = 1;
	              int mu = snprintf(RcvData, sizeof(RcvData),
	                                "UPSTART,%lu,%.3f\r\n",
	                                (unsigned long)t_upstart_us, (double)IIR_Output_PPG);
	              HAL_UART_Transmit(&huart2, (uint8_t*)RcvData, (uint16_t)mu, 20); // ← '(직접 반영해)' 같은 텍스트 넣지 마세요
	              st = ST_RISING;
	          }
	      }

	      // RISING: 임계 이상에서 기울기 최대가 나오면 SLOPEMAXED로
	      if (st == ST_RISING) {
	          // SLOPEMAX: 데이터>thr, der1>0 유지, 2차: 양→음
	          if ((IIR_Output_PPG > thr) &&
	              (der1 > 0.0f) &&
	              (prevDer2 > 0.0f) && (der2 <= 0.0f))  // ★ 양→음
	          {
	              t_slopemax_us = now_us;
	              slopemax_hit = 1;
	              int ms = snprintf(RcvData, sizeof(RcvData),
	                                "SLOPEMAX,%lu,%.3f\r\n",
	                                (unsigned long)t_slopemax_us, (double)IIR_Output_PPG);
	              HAL_UART_Transmit(&huart2, (uint8_t*)RcvData, (uint16_t)ms, 20);
	              st = ST_SLOPEMAXED;
	          }
	      }

	      // SLOPEMAXED: 임계 이상에서 1차가 양→음이면 PEAK, IBI 산출, 다시 IDLE
	      if (st == ST_SLOPEMAXED) {
	          if ((IIR_Output_PPG > thr) &&
	              (prevDer1 > 0.0f && der1 <= 0.0f) &&
	              ((lastPeak_us == 0) || (dt_us(now_us, lastPeak_us) > REFRACTORY_US)))
	          {
	              // IBI 계산
	              if (lastPeak_us != 0) {
	                  uint32_t ibi_cur_ms = dt_us(now_us, lastPeak_us) / 1000U;
	                  if (ibi_cur_ms >= IBI_MIN_MS && ibi_cur_ms <= IBI_MAX_MS) {
	                      currentIBI_ms = ibi_cur_ms;
	                      ibiValid = 1;
	                      // ★ 여기 추가
	                      last_valid_ibi_ms = ibi_cur_ms;

	                      if (ibiIndex < MAX_IBI) {
	                          ibi_ms[ibiIndex++] = ibi_cur_ms;
	                      } else {
	                          memmove(&ibi_ms[0], &ibi_ms[1], (MAX_IBI - 1) * sizeof(uint32_t));
	                          ibi_ms[MAX_IBI - 1] = ibi_cur_ms;
	                      }
	                  }
	              }
	              lastPeak_us = now_us;
	              t_peak_us = now_us;
	              peak_hit = 1;

	              // --- PEAK 블록 안 ---
	              int n1 = snprintf(RcvData, sizeof(RcvData), "%.3f,%lu\r\n",
	                                (double)IIR_Output_PPG,
	                                (unsigned long)(ibiValid ? currentIBI_ms : 0UL));
	              if (n1 > 0) HAL_UART_Transmit(&huart2, (uint8_t*)RcvData, (uint16_t)n1, 20);

	              // 피크에서만 표시했음을 표시하고 끝. 여기서는 절대 ,0를 출력/초기화하지 않음
	              just_peaked = 1;

	              st = ST_IDLE;   // 한 박동 종료

	          }
	      }

	      // ---- 스트림(그래프용): PPG, IBI(ms) ----
	      // 이번 샘플에서 피크가 없었다면 PPG,0 한 줄 출력
	      if (!just_peaked) {
	          int n = snprintf(RcvData, sizeof(RcvData), "%.3f,0\r\n",
	                           (double)IIR_Output_PPG);
	          if (n > 0) HAL_UART_Transmit(&huart2, (uint8_t*)RcvData, (uint16_t)n, 20);
	      }

	      // 상태 업데이트 (반드시 마지막에 플래그 클리어)
	      prevDer2 = der2;
	      prevDer1 = der1;
	      prevPPG  = IIR_Output_PPG;
	      just_peaked = 0;

	      // 샘플 주파수/PC 부하 고려해 delay 유지 (필요시 조정)
	      HAL_Delay(5);
  }

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ENABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = ENABLE;
  hadc1.Init.NbrOfDiscConversion = 1;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 2;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_56CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Rank = 2;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 999;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_ENABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 83;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
// ✅ 여기에 넣으면 됩니다.
static inline uint32_t t_us(void){
    return __HAL_TIM_GET_COUNTER(&htim2);
}

static inline uint32_t dt_us(uint32_t now, uint32_t then){
    return (now - then);  // 오버플로 자동 보정
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */  >> stm인데 라즈베리 4에서 실행할거임. peak뜰때마다 LED 불들어오게하고, 심박수는 8segment display에 숫자 뜨게할거임