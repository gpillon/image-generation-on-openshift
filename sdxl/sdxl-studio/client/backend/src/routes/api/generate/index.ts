import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import axios from 'axios';
import WebSocket from 'ws';
import {
  getSDXLEndpoint,
  getGuardEnabled,
  getSafetyCheckEnabled,
  getGuardConfig,
  getSafetyCheckConfig,
  updateLastActivity,
  getFluxEndpoint,
  getWanEndpoint,
} from '../../../utils/config'; // Adjust the import path as needed
import guard from '../../../services/guard';
import safetyChecker from '../../../services/image-safety-check';
import { Payload } from '../../../schema/payload';
import { safeImage } from '../../../utils/safeImage';

export default async (fastify: FastifyInstance): Promise<void> => {
  const decoder = new TextDecoder('utf-8');
  const jobTracker: Payload[] = [];

  // ============================
  // 1. POST Endpoint: Start Job
  // ============================
  fastify.post('/', async (req: FastifyRequest, reply: FastifyReply) => {
    const {
      prompt,
      guidance_scale,
      num_inference_steps,
      crops_coords_top_left,
      width,
      height,
      model,
      denoising_limit,
      num_frames,
      fps,
    } = req.body as any;

    const data: Payload = {
      prompt,
      guidance_scale,
      num_inference_steps,
      crops_coords_top_left,
      width,
      height,
      denoising_limit,
      past_threshold: false,
      image_failed_check: false,
    };

    // Add video-specific parameters when needed
    if (model === 'wan' && num_frames) {
      data.num_frames = num_frames || 80;
    }
    if (model === 'wan' && fps) {
      data.fps = fps || 16;
    }
    if (model === 'wan' && num_inference_steps) {
      data.num_inference_steps = num_inference_steps || 50;
    }

    updateLastActivity();

    let modelEndpoint;
    switch (model) {
      case 'sdxl':
        modelEndpoint = getSDXLEndpoint();
        break;
      case 'flux':
        modelEndpoint = getFluxEndpoint();
        break;
      case 'wan':
        modelEndpoint = getWanEndpoint();
        break;
      default:
        reply.code(400).send({ message: 'Invalid model' });
        return;
    }

    const guardConfig = getGuardConfig();
    const safetyCheckConfig = getSafetyCheckConfig();

    // Check all is well with the environment configurations
    if (getGuardEnabled() === 'true') {
      if (guardConfig.guardEndpointToken === '' || guardConfig.guardEndpointURL === '') {
        reply.code(403).send({
          message: 'Guardrails not configured correctly',
        });
        return;
      }
    }

    if (getSafetyCheckEnabled() === 'true') {
      if (
        safetyCheckConfig.safetyCheckEndpointToken === '' ||
        safetyCheckConfig.safetyCheckEndpointURL === ''
      ) {
        reply.code(403).send({
          message: 'Safety checker not configured correctly',
        });
        return;
      }
    }

    if (getGuardEnabled() === 'true') {
      const failedGuardCheck = await guard(data);
      if (failedGuardCheck) {
        reply.code(403).send({
          message:
            'Your query appears to contain inappropriate content. Please rephrase and try again',
        });
        return;
      }
    }

    console.log(`Sending request to ${model} endpoint:`, modelEndpoint.endpointURL + '/generate');

    const response = await axios.post(
      modelEndpoint.endpointURL + `/generate?user_key=${modelEndpoint.endpointToken}`,
      data,
    );

    const { job_id } = response.data;
    if (!job_id) {
      reply.code(500).send({ message: 'No job_id returned from generation endpoint.' });
      return;
    }
    jobTracker[parseInt(job_id)] = data;
    reply.send({ job_id, model });
  });

  // =======================================================
  // 2. WebSocket Endpoint: Pipe API updates to the Client
  // =======================================================
  fastify.get('/progress/:job_id', { websocket: true }, (connection, req) => {
    const { job_id } = req.params as { job_id: string };
    console.log(`Client connected for job_id: ${job_id}`);

    const { model } = req.query as { model: string };

    let modelEndpoint;
    switch (model) {
      case 'sdxl':
        modelEndpoint = getSDXLEndpoint();
        break;
      case 'flux':
        modelEndpoint = getFluxEndpoint();
        break;
      case 'wan':
        modelEndpoint = getWanEndpoint();
        break;
      default:
        connection.socket.send(JSON.stringify({ error: 'Invalid model' }));
        connection.socket.close();
        return;
    }
    // Connect to the external API WebSocket
    const wsProtocol = modelEndpoint.endpointURL.startsWith('https') ? 'wss' : 'ws';
    const backendHost = modelEndpoint.endpointURL.replace(/^https?:\/\//, '');

    const apiWsUrl = `${wsProtocol}://${backendHost}/progress/${job_id}?user_key=${modelEndpoint.endpointToken}`;
    console.log(`Connecting to API WebSocket: ${apiWsUrl}`);
    const apiWs = new WebSocket(apiWsUrl);

    // Handle incoming messages from the API WebSocket
    apiWs.on('message', async (data: WebSocket.Data) => {
      try {
        const dataString = typeof data === 'string' ? data : decoder.decode(data as ArrayBuffer);
        const msg = JSON.parse(dataString);
        msg.job_id = job_id;

        // Get the current job details from the jobTracker array
        const currentJob = jobTracker[parseInt(job_id)];

        // Check to see if we've passed the threshold for image checking, and if we have not, check if we need to set this on the jobTracker array.
        if (!currentJob.past_threshold && currentJob.num_inference_steps / 2 < msg.step) {
          jobTracker[parseInt(job_id)].past_threshold = true;
        }

        if (getSafetyCheckEnabled() === 'true' && msg.image && currentJob.past_threshold) {
          const failedSafetyCheck = await safetyChecker(msg.image);
          if (!failedSafetyCheck) {
            // Forward the message to the client
            connection.socket.send(JSON.stringify(msg));
          } else {
            // The image failed the safety check, replace with the safe image.
            msg.image_failed_check = true;
            currentJob.image_failed_check = true;
            msg.image = safeImage;
            connection.socket.send(JSON.stringify(msg));
          }
        } else {
          // Forward the message to the client
          connection.socket.send(JSON.stringify(msg));
        }
        // Close both connections when the job is complete
        if (msg.status === 'completed') {
          apiWs.close();
          connection.socket.close();
        }
      } catch (err) {
        console.error('Error parsing message from API:', err);
      }
    });

    // Handle API WebSocket errors
    apiWs.on('error', (error) => {
      console.error('API WebSocket error:', error);
      connection.socket.send(JSON.stringify({ error: 'Error receiving job updates.' }));
      connection.socket.close();
    });

    // When API WebSocket closes, close the client WebSocket
    apiWs.on('close', () => {
      console.log(`API WebSocket closed for job_id: ${job_id}`);
      if (connection.socket.readyState === WebSocket.OPEN) {
        connection.socket.close();
      }
    });

    // Handle client messages (optional)
    connection.socket.on('message', (data) => {
      const dataString = typeof data === 'string' ? data : decoder.decode(data as ArrayBuffer);
      console.log('Received message from client:', dataString);
    });

    // When the client disconnects, close the API WebSocket if open
    connection.socket.on('close', () => {
      console.log(`Client WebSocket closed for job_id: ${job_id}`);
      if (apiWs.readyState === WebSocket.OPEN) {
        apiWs.close();
      }
    });
  });

  // =======================================================
  // 3. Video Download Endpoint: Get the generated video
  // =======================================================
  fastify.get('/video/:job_id', async (req: FastifyRequest, reply: FastifyReply) => {
    const { job_id } = req.params as { job_id: string };
    const { model } = req.query as { model: string };
    
    if (model !== 'wan') {
      reply.code(400).send({ message: 'Invalid model for video download' });
      return;
    }

    try {
      const modelEndpoint = getWanEndpoint();
      
      // Forward the video request to the WAN endpoint
      const response = await axios({
        method: 'get',
        url: `${modelEndpoint.endpointURL}/video/${job_id}?user_key=${modelEndpoint.endpointToken}`,
        responseType: 'stream'
      });
      
      // Forward the stream response
      reply.header('Content-Type', 'video/mp4');
      reply.header('Content-Disposition', `attachment; filename=video_${job_id}.mp4`);
      
      return response.data;
    } catch (error) {
      console.error('Error fetching video:', error);
      reply.code(500).send({ message: 'Error fetching video' });
    }
  });
};
