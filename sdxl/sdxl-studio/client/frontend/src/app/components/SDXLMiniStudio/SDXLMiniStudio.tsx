import config from '@app/config';
import { ActionGroup, Button, Card, CardBody, CardTitle, Flex, FlexItem, Form, FormGroup, FormSelect, FormSelectOption, Page, PageSection, Popover, Progress, ProgressMeasureLocation, Slider, SliderOnChangeEvent, Text, TextArea, TextContent, TextVariants } from '@patternfly/react-core';
import { HelpIcon } from '@patternfly/react-icons';
import styles from '@patternfly/react-styles/css/components/Form/form';
import axios from 'axios';
import * as React from 'react';
import Emitter from '../../utils/emitter';
import DocumentRenderer from '../DocumentRenderer/DocumentRenderer';

interface SDXLMiniStudioProps { }

class GenerateParameters {
  static denoising_limit_models: string[] = ['sdxl'];
  static num_frames_models: string[] = ['wan'];
  static fps_models: string[] = ['wan'];
  static video_models: string[] = ['wan'];
  static image_models: string[] = ['sdxl', 'flux'];

  prompt: string;
  guidance_scale: number;
  num_inference_steps: number;
  crops_coords_top_left: number[];
  width: number;
  height: number;
  denoising_limit: number;
  model: string;
  num_frames?: number;
  fps?: number;


  constructor(
    prompt: string = 'A cat with a red fedora holding a sign that says "Flux Studio"',
    guidance_scale: number = 8.0,
    num_inference_steps: number = 40,
    crops_coords_top_left: number[] = [0, 0],
    model: string = 'sdxl',
    width: number = 1024,
    height: number = 1024,
    denoising_limit: number = 0.8,
    num_frames: number = 80,
    fps: number = 15
  ) {
    this.prompt = prompt;
    this.guidance_scale = guidance_scale;
    this.num_inference_steps = num_inference_steps;
    this.crops_coords_top_left = crops_coords_top_left;
    this.width = width;
    this.height = height;
    this.denoising_limit = denoising_limit;
    this.model = model;

    // Only add video params for WAN model
    if (GenerateParameters.video_models.includes(model)) {
      this.num_frames = num_frames;
      this.fps = fps;
    }
  }
}

const SDXLMiniStudio: React.FunctionComponent<SDXLMiniStudioProps> = () => {

  const [imagePanelTitle, setImagePanelTitle] = React.useState('Instructions');
  const [documentRendererVisible, setDocumentRendererVisible] = React.useState(false);
  const [progressVisible, setProgressVisible] = React.useState(false);

  const [generateParameters, setGenerateParameters] = React.useState<GenerateParameters>(new GenerateParameters());
  const handleGenerateParametersChange = (value, field) => {
    setGenerateParameters(prevState => ({
      ...prevState,
      [field]: value,
    }));;
  }

  const [prompt, setPrompt] = React.useState('');
  const handlePromptChange = (_event: React.FormEvent<HTMLTextAreaElement>, value: string) => {
    setPrompt(value);
    handleGenerateParametersChange(value, 'prompt');
  }

  const sizeOptions = [
    { value: 'standard', label: 'Standard, 1:1, 1024 x 1024', models: ['sdxl', 'flux'] },
    { value: 'small', label: 'Small, 1:1, 512 x 512', models: ['sdxl', 'flux'] },
    { value: 'vertical', label: 'Vertical, 4:7, 768 x 1344', models: ['sdxl', 'flux'] },
    { value: 'portrait', label: 'Portrait, 7:9, 896 x 1152', models: ['sdxl', 'flux'] },
    { value: 'photo', label: 'Photo, 9:7, 1152 x 896', models: ['sdxl', 'flux'] },
    { value: 'landscape', label: 'Landscape, 19:13, 1216 x 832', models: ['sdxl', 'flux'] },
    { value: 'widescreen', label: 'Widescreen, 7:4, 1344 x 768', models: ['sdxl', 'flux'] },
    { value: 'cinematic', label: 'Cinematic, 12:5, 1536 x 640', models: ['sdxl', 'flux'] },
    { value: 'standard-video', label: 'Standard, 1:1, 832 x 480', models: ['wan'] },
    { value: 'square-video', label: 'Square, 1:1, 480 x 480', models: ['wan'] },
  ];

  const modelOptions = [
    { value: 'sdxl', label: 'SDXL' },
    { value: 'flux', label: 'Flux' },
    { value: 'wan', label: 'WAN (Video)' },
  ];

  const [sizeOption, setSizeOption] = React.useState('standard');
  const handleSizeOptionChange = (_event: React.FormEvent<HTMLSelectElement>, value: string) => {
    switch (value) {
      case 'small':
        handleGenerateParametersChange(512, 'width');
        handleGenerateParametersChange(512, 'height');
        break;
      case 'standard':
        handleGenerateParametersChange(1024, 'width');
        handleGenerateParametersChange(1024, 'height');
        break;
      case 'vertical':
        handleGenerateParametersChange(768, 'width');
        handleGenerateParametersChange(1344, 'height');
        break;
      case 'portrait':
        handleGenerateParametersChange(896, 'width');
        handleGenerateParametersChange(1152, 'height');
        break;
      case 'photo':
        handleGenerateParametersChange(1152, 'width');
        handleGenerateParametersChange(896, 'height');
        break;
      case 'landscape':
        handleGenerateParametersChange(1216, 'width');
        handleGenerateParametersChange(832, 'height');
        break;
      case 'widescreen':
        handleGenerateParametersChange(1344, 'width');
        handleGenerateParametersChange(768, 'height');
        break;
      case 'cinematic':
        handleGenerateParametersChange(1536, 'width');
        handleGenerateParametersChange(640, 'height');
        break;
      case 'standard-video':
        handleGenerateParametersChange(832, 'width');
        handleGenerateParametersChange(480, 'height');
        break;
      case 'square-video':
        handleGenerateParametersChange(480, 'width');
        handleGenerateParametersChange(480, 'height');
        break;
    }
    setSizeOption(value);
  }

  const [guidance_scale, setGuidanceScale] = React.useState(8.0);
  const handleGuidanceScaleChange = (_event: SliderOnChangeEvent, value: number) => {
    setGuidanceScale(value);
    handleGenerateParametersChange(value, 'guidance_scale');
  }

  const [modelOption, setModelOption] = React.useState('sdxl');
  const handleModelOptionChange = (_event: React.FormEvent<HTMLSelectElement>, value: string) => {
    setModelOption(value);
    handleGenerateParametersChange(value, 'model');
    switch (value) {
      case 'sdxl':
        setNumInferenceSteps(40)
        handleGenerateParametersChange(40, 'num_inference_steps');
        break;
      case 'flux':
        setNumInferenceSteps(4)
        handleGenerateParametersChange(4, 'num_inference_steps');
        break;
      case 'wan':
        setNumInferenceSteps(50)
        handleGenerateParametersChange(50, 'num_inference_steps');
        // Set default video params
        setNumFrames(40);
        handleGenerateParametersChange(40, 'num_frames');
        setFps(16);
        handleGenerateParametersChange(16, 'fps');
        // For videos, use landscape format
        setSizeOption('standard-video');
        handleGenerateParametersChange(832, 'width');
        handleGenerateParametersChange(480, 'height');
        break;
    }
  }

  const [num_inference_steps, setNumInferenceSteps] = React.useState(40);
  const handleNumInferenceStepsChange = (_event: SliderOnChangeEvent, value: number) => {
    setNumInferenceSteps(value);
    handleGenerateParametersChange(value, 'num_inference_steps');
  }

  const [denoising_limit, setDenoisingLimit] = React.useState(80);
  const handleDenoisingLimitChange = (_event: SliderOnChangeEvent, value: number) => {
    setDenoisingLimit(value);
    handleGenerateParametersChange(value / 100, 'denoising_limit');
  }

  const [fileData, setFileData] = React.useState('');
  const [fileName, setFileName] = React.useState('');
  const [videoUrl, setVideoUrl] = React.useState('');
  const [phase, setPhase] = React.useState('Base');
  const [baseStep, setBaseStep] = React.useState(0);
  const [refinerStep, setRefinerStep] = React.useState(0);

  const [num_frames, setNumFrames] = React.useState(80);
  const handleNumFramesChange = (_event: SliderOnChangeEvent, value: number) => {
    setNumFrames(value);
    handleGenerateParametersChange(value, 'num_frames');
  }

  const [fps, setFps] = React.useState(15);
  const handleFpsChange = (_event: SliderOnChangeEvent, value: number) => {
    setFps(value);
    handleGenerateParametersChange(value, 'fps');
  }

  // Add a new state variable for progress percentage
  const [progressPct, setProgressPct] = React.useState(0);

  const handleGenerateImage = (event) => {
    event.preventDefault();
    setImagePanelTitle('Sending generation request');
    setDocumentRendererVisible(true);
    setVideoUrl('');
    Emitter.emit('notification', {
      variant: 'success',
      title: '',
      description: 'Generation request sent! Please wait...',
    });

    axios
      .post(`${config.backend_api_url}/generate`, generateParameters)
      .then((response) => {
        // Extract the job_id from the response.
        const { job_id, model } = response.data;
        if (!job_id) {
          Emitter.emit('notification', {
            variant: 'warning',
            title: '',
            description: 'No job_id received from backend!',
          });
          if (!model) {
            Emitter.emit('notification', {
              variant: 'warning',
              title: '',
              description: 'No model received from backend!',
            });
          }
          return;
        }

        // Create the WebSocket URL derived from backend API.
        const wsProtocol = config.backend_api_url.startsWith('https') ? 'wss' : 'ws';
        const backendHost = config.backend_api_url.replace(/^https?:\/\//, '');
        const wsUrl = `${wsProtocol}://${backendHost}/generate/progress/${job_id}?model=${model}`;

        // Open a WebSocket connection.
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log('WebSocket connected for job_id:', job_id);
        };

        ws.onmessage = (messageEvent) => {
          try {
            const msg = JSON.parse(messageEvent.data);

            // If this is a queue update, update queue info.
            if (msg.status && msg.status === 'queued') {
              if (msg.position === -1) {
                setImagePanelTitle(GenerateParameters.video_models.includes(model) ? 'Video generation is starting...' : 'Image generation is starting...');
              } else if (msg.position === 1) {
                setImagePanelTitle(GenerateParameters.video_models.includes(model) ? 'You are next! (#1 in the queue)' : 'You are next! (#1 in the queue)');
              }
              else {
                setImagePanelTitle('Currently serving others, you are #' + msg.position + ' in the queue.');
              }
              setProgressVisible(false);
              setFileName('');
              setFileData('');
              setVideoUrl('');
            }

            // For WAN model, handle video ready notification
            if (msg.status && msg.status === 'video_ready') {
              setVideoUrl(`${config.backend_api_url}/generate/video/${job_id}?model=${model}`);
            }

            // If this is a progress update, update UI elements.
            if (msg.status && msg.status === 'progress') {
              setImagePanelTitle((GenerateParameters.video_models.includes(model) ? 'Video' : 'Image') + ' generation in progress: ' + msg.progress + '%');
              setProgressVisible(true);
              setProgressPct(msg.progress || 0);

              if (msg.pipeline && msg.pipeline === 'base') {
                setPhase('Base');
                setBaseStep(msg.step + 1);
              }
              if (msg.pipeline && msg.pipeline === 'refiner') {
                setPhase('Refiner');
                setRefinerStep(msg.step + 1);
              }

              setFileName('new_image.png');
              setFileData(msg.image);
            }

            // If the job is complete, update the image data and close the WebSocket.
            if (msg.status && msg.status === 'completed') {
              if (msg.image_failed_check === true) {
                setProgressVisible(false);
                setBaseStep(0);
                setRefinerStep(0);
                setImagePanelTitle('Image generation error');
                Emitter.emit('notification', {
                  variant: 'warning',
                  title: '',
                  description: 'Sorry, the generated image has been classified as sensitive and blocked!',
                });
              } else {
                setProgressVisible(false);
                setBaseStep(0);
                setRefinerStep(0);
                setFileName('new_image.png');
                setFileData(msg.image);

                // Handle warnings if present
                if (msg.warning) {
                  setImagePanelTitle(GenerateParameters.video_models.includes(model) ? 'Video generated with warnings' : 'Image generated with warnings');
                  Emitter.emit('notification', {
                    variant: 'warning',
                    title: '',
                    description: GenerateParameters.video_models.includes(model)
                      ? 'Video generated but with warnings. You can still download the video.'
                      : (msg.warning || 'Generation completed with warnings'),
                  });
                } else {
                  setImagePanelTitle((GenerateParameters.video_models.includes(model) ? 'Video' : 'Image') + ' generated in ' + msg.processing_time.toFixed(1) + ' seconds');
                  Emitter.emit('notification', {
                    variant: 'success',
                    title: '',
                    description: GenerateParameters.video_models.includes(model)
                      ? 'Video generated! Check the Video Output panel.'
                      : (msg.description || 'Image generated!'),
                  });
                }
              }

              ws.close();
            }
          } catch (err) {
            console.error('Error parsing message from WebSocket:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          Emitter.emit('notification', {
            variant: 'warning',
            title: '',
            description: 'WebSocket error occurred.',
          });
        };

        ws.onclose = () => {
          console.log('WebSocket connection closed.');
        };
      })
      .catch((error) => {
        setDocumentRendererVisible(false);
        Emitter.emit('notification', {
          variant: 'warning',
          title: '',
          description:
            'Connection failed with the error: ' +
            (error.response && error.response.data && error.response.data.message
              ? error.response.data.message
              : error.message),
        });
      });
  };


  const handleReset = (event) => {
    event.preventDefault();
    setDocumentRendererVisible(false);
    setGenerateParameters(new GenerateParameters());
    setPrompt('');
    setImagePanelTitle('Instructions');
    setSizeOption('standard');
    setGuidanceScale(8.0);
    setNumInferenceSteps(40);
    setDenoisingLimit(80);
    setBaseStep(0);
    setRefinerStep(0);
    setFileData('');
    setFileName('');
    setVideoUrl('');
    setNumFrames(80);
    setFps(16);
  }

  return (
    <Page className='sdxl-ministudio'>
      <PageSection>
        <TextContent>
          <Text component={TextVariants.h1}>SDXL & Video Generation Studio</Text>
        </TextContent>
      </PageSection>
      <PageSection>
        <Flex>
          <FlexItem flex={{ default: 'flex_1' }}>
            <Card component="div">
              <CardTitle>Parameters</CardTitle>
              <CardBody>
                <Form onSubmit={handleGenerateImage}>
                  <FormGroup
                    label="Prompt"
                    fieldId="prompt"
                    labelIcon={
                      <Popover
                        headerContent={
                          <div>
                            The description of the image to generate.
                          </div>
                        }
                        bodyContent={
                          <div>
                            <p>Describe what you want to generate.</p>
                            <p>For example, "A beautiful sunset over the ocean with a sailboat in the distance".</p>
                            <p>Here is a <a href="https://blog.segmind.com/prompt-guide-for-stable-diffusion-xl-crafting-textual-descriptions-for-image-generation/" target="_blank" rel="noreferrer">full guide</a> for Stable Diffusion XL prompting.</p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-02"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <TextArea
                      value={prompt}
                      id="prompt"
                      name="prompt"
                      aria-label="prompt"
                      placeholder="Describe what you want to generate"
                      onChange={handlePromptChange}
                    />
                  </FormGroup>
                  <FormGroup
                    label="Model"
                    fieldId="model"
                    labelIcon={
                      <Popover
                        headerContent={
                          <div>
                            The model to use for image generation.
                          </div>
                        }
                        bodyContent={
                          <div>
                            <p>SDXL or Flux </p>
                            <p>Select one.</p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-03"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <FormSelect
                      value={modelOption}
                      id="model"
                      name="model"
                      aria-label="model"
                      onChange={handleModelOptionChange}
                    >
                      {modelOptions.map((option, index) => (
                        <FormSelectOption key={index} value={option.value} label={option.label} />
                      ))}
                    </FormSelect>
                  </FormGroup>
                  <FormGroup
                    label="Size"
                    fieldId="size"
                    labelIcon={
                      <Popover
                        headerContent={
                          <div>
                            The size of the image to generate.
                          </div>
                        }
                        bodyContent={
                          <div>
                            <p>SDXL can only generate images of predefined sizes.</p>
                            <p>Select one.</p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-02"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <FormSelect
                      value={sizeOption}
                      id="size"
                      name="size"
                      aria-label="size"
                      onChange={handleSizeOptionChange}
                    >
                      {sizeOptions.filter(option => option.models.includes(modelOption)).map((option, index) => (
                        <FormSelectOption key={index} value={option.value} label={option.label} />
                      ))}
                    </FormSelect>
                  </FormGroup>
                  <FormGroup
                    label={`Guidance Scale: ${guidance_scale}`}
                    fieldId="guidance_scale"
                    labelIcon={
                      <Popover
                        bodyContent={
                          <div>
                            <p><b>Guidance scale</b> is a parameter that controls the balance between adhering to the provided text prompt and the inherent "creativity" or randomness of the model.</p>
                            <p>
                              <ul>
                                <li><b>Low guidance scale</b>: The model has more creative freedom, and the resulting image may not closely match the prompt but can introduce unexpected details.</li>
                                <li><b>High guidance scale</b>: The model is more constrained by the prompt, leading to images that align more strictly with the given description but may be less varied or imaginative.</li>
                              </ul>
                            </p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-02"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <Slider
                      id="guidance_scale"
                      value={guidance_scale}
                      onChange={handleGuidanceScaleChange}
                      aria-labelledby="Guidance Scale"
                      hasTooltipOverThumb
                      min={0}
                      max={20}
                      step={0.5}
                    />
                  </FormGroup>
                  <FormGroup
                    label={`Number of Inference Steps: ${num_inference_steps}`}
                    fieldId="num_inference_steps"
                    labelIcon={
                      <Popover
                        bodyContent={
                          <div>
                            <p>
                              <b>Number of inference steps</b> is a parameter that controls the number of steps the model takes to generate an image.
                            </p>
                            <p>
                              <ul>
                                <li><b>Low number of inference steps</b>: The model generates images more quickly but may produce lower-quality results.</li>
                                <li><b>High number of inference steps</b>: The model generates images more slowly but may produce higher-quality results.</li>
                              </ul>
                            </p>
                            <p>
                              A minimum of <b>30 steps</b> is recommended for high-quality results for SDXL.
                            </p>
                            <p>
                              For Flux, a minimum of <b>3 steps</b> is recommended for medium-quality results. <b>4 steps</b> is recommended for high-quality results.
                            </p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-02"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <Slider
                      id="num_inference_steps"
                      value={num_inference_steps}
                      onChange={handleNumInferenceStepsChange}
                      aria-labelledby="Number of Inference Steps"
                      hasTooltipOverThumb
                      min={2}
                      max={100}
                      step={1}
                    />
                  </FormGroup>
                  {GenerateParameters.denoising_limit_models.includes(modelOption) && (
                    <FormGroup
                      label={`Denoising Limit: ${denoising_limit} %`}
                      fieldId="denoising_limit"
                    labelIcon={
                      <Popover
                        bodyContent={
                          <div>
                            <p>
                              Stable Diffusion XL uses two models: a diffusion model to generate the image and a denoising model to refine it.
                            </p>
                            <p>
                              The <b>denoising limit</b> parameter controls when the generation model passes over to the refining model.
                            </p>
                            <p>
                              With 40 steps and a denoising limit at 80%, 32 steps will be used for generation, and 8 for refinement.
                            </p>
                          </div>
                        }
                      >
                        <button
                          type="button"
                          aria-label="More info for name field"
                          onClick={(e) => e.preventDefault()}
                          aria-describedby="simple-form-name-02"
                          className={styles.formGroupLabelHelp}
                        >
                          <HelpIcon />
                        </button>
                      </Popover>
                    }>
                    <Slider
                      id="denoising_limit"
                      value={denoising_limit}
                      onChange={handleDenoisingLimitChange}
                      areCustomStepsContinuous
                      aria-labelledby="Denoising Limit"
                      hasTooltipOverThumb
                      min={1}
                      max={99}
                      step={1}
                      />
                    </FormGroup>
                  )}
                  {GenerateParameters.num_frames_models.includes(modelOption) && (
                    <>
                      <FormGroup
                        label="Number of frames"
                        fieldId="num_frames"
                        labelIcon={
                          <Popover
                            headerContent={<div>Number of frames to generate</div>}
                            bodyContent={
                              <div>
                                <p>Controls the duration of the generated video. More frames means a longer video.</p>
                                <p>Default is 80 frames which produces a ~5 second video at 15 fps.</p>
                              </div>
                            }
                          >
                            <button
                              type="button"
                              aria-label="More info for frames field"
                              onClick={(e) => e.preventDefault()}
                              className={`${styles.formGroupLabelHelp}`}
                            >
                              <HelpIcon />
                            </button>
                          </Popover>
                        }
                      >
                        <div>
                          <Slider
                            id="num_frames_slider"
                            min={16}
                            max={160}
                            value={num_frames}
                            onChange={handleNumFramesChange}
                            showBoundaries
                            hasTooltipOverThumb
                            showTicks
                            step={4}
                          />
                        </div>
                      </FormGroup>
                      <FormGroup
                        label="Frames per second (fps)"
                        fieldId="fps"
                        labelIcon={
                          <Popover
                            headerContent={<div>FPS (Frames per second)</div>}
                            bodyContent={
                              <div>
                                <p>Controls how fast the generated video will play.</p>
                                <p>Lower values create slower videos, higher values create faster videos.</p>
                              </div>
                            }
                          >
                            <button
                              type="button"
                              aria-label="More info for fps field"
                              onClick={(e) => e.preventDefault()}
                              className={`${styles.formGroupLabelHelp}`}
                            >
                              <HelpIcon />
                            </button>
                          </Popover>
                        }
                      >
                        <div>
                          <Slider
                            id="fps_slider"
                            min={4}
                            max={32}
                            value={fps}
                            onChange={handleFpsChange}
                            showBoundaries
                            hasTooltipOverThumb
                            showTicks
                            step={4}
                          />
                        </div>
                      </FormGroup>
                    </>
                  )}
                  <ActionGroup>
                    <Button type="submit" variant="primary">
                      {GenerateParameters.video_models.includes(modelOption) ? 'Generate the video' : 'Generate the image'}
                    </Button>
                    <Button type="reset" variant="secondary" onClick={handleReset}>Reset</Button>
                  </ActionGroup>
                </Form>
              </CardBody>
            </Card>
          </FlexItem>
          <FlexItem flex={{ default: 'flex_3' }}>
            <Card component="div">
              <CardTitle>
                {GenerateParameters.video_models.includes(modelOption) ? 'Video Output' : 'Image Output'}: {imagePanelTitle}
              </CardTitle>
              <CardBody>
                {progressVisible &&
                  <Progress
                    title={GenerateParameters.video_models.includes(modelOption) ? "Video Generation Progress" : "Image Generation Progress"}
                    value={progressPct}
                    valueText={`${progressPct}% - ${phase} phase`}
                    label={`Step ${baseStep + refinerStep} of ${num_inference_steps}`}
                    max={100}
                    measureLocation={ProgressMeasureLocation.top}
                    style={{ paddingBottom: '1rem' }}
                  />
                }
                {!documentRendererVisible && <p>Enter the description of the image to generate in the prompt, adjust the parameters if you want, and click on <b>Generate the image</b>.</p>}
                {documentRendererVisible && !videoUrl && <DocumentRenderer fileData={fileData} fileName={fileName} width={generateParameters.width} height={generateParameters.height} />}
                {videoUrl && (
                  <div style={{ marginTop: '20px' }}>
                    <h3>Generated Video</h3>
                    <video
                      controls
                      width="100%"
                      src={videoUrl}
                      style={{ maxHeight: '600px', objectFit: 'contain' }}
                    >
                      Your browser does not support the video tag.
                    </video>
                    <Button
                      variant="secondary"
                      component="a"
                      href={videoUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      download
                    >
                      Download Video
                    </Button>
                  </div>
                )}
              </CardBody>
            </Card>
          </FlexItem>
        </Flex>
      </PageSection>
    </Page>

  )
};

export default SDXLMiniStudio;
