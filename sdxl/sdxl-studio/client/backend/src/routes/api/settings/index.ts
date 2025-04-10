import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';

import {
  getSDXLEndpoint,
  setSDXLEndpoint,
  getParasolMode,
  updateLastActivity,
  getFluxEndpoint,
  setFluxEndpoint,
  getWanEndpoint,
  setWanEndpoint,
} from '../../../utils/config';
import axios from 'axios';

export default async (fastify: FastifyInstance): Promise<void> => {
  // Retrieve endpoint settings
  fastify.get('/sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const endpointUrl = getSDXLEndpoint().endpointURL;
    const endpointToken = getSDXLEndpoint().endpointToken;
    const fluxEndpointUrl = getFluxEndpoint().endpointURL;
    const fluxEndpointToken = getFluxEndpoint().endpointToken;
    const wanEndpointUrl = getWanEndpoint().endpointURL;
    const wanEndpointToken = getWanEndpoint().endpointToken;

    const settings = {
      endpointUrl: endpointUrl,
      endpointToken: endpointToken,
      fluxEndpointUrl: fluxEndpointUrl,
      fluxEndpointToken: fluxEndpointToken,
      wanEndpointUrl: wanEndpointUrl,
      wanEndpointToken: wanEndpointToken,
    };
    console.log(settings);
    reply.send({ settings });
  });

  // Update endpoint settings
  fastify.put('/sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const { endpointUrl, endpointToken, fluxEndpointUrl, fluxEndpointToken, wanEndpointUrl, wanEndpointToken } = req.body as any;
    setSDXLEndpoint(endpointUrl, endpointToken);
    setFluxEndpoint(fluxEndpointUrl, fluxEndpointToken);
    setWanEndpoint(wanEndpointUrl, wanEndpointToken);
    reply.send({ message: 'Settings updated successfully!' });
  });

  // Test endpoint connection
  fastify.post('/test-sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const { endpointUrl, endpointToken, fluxEndpointUrl, fluxEndpointToken, wanEndpointUrl, wanEndpointToken } = req.body as any;
    try {
      const promises = [];
      if (endpointUrl) {
        promises.push(axios.get(endpointUrl + '/health?user_key=' + endpointToken));
      } else {
        promises.push(Promise.resolve({ status: 200 }));
      }
      if (fluxEndpointUrl) {
        promises.push(axios.get(fluxEndpointUrl + '/health?user_key=' + fluxEndpointToken));
      } else {
        promises.push(Promise.resolve({ status: 200 }));
      }
      if (wanEndpointUrl) {
        promises.push(axios.get(wanEndpointUrl + '/health?user_key=' + wanEndpointToken));
      } else {
        promises.push(Promise.resolve({ status: 200 }));
      }
      const [sdxlResponse, fluxResponse, wanResponse] = await Promise.all(promises);
      if (sdxlResponse.status === 200 && fluxResponse.status === 200 && wanResponse.status === 200) {
        reply.send({
          message: 'Connection successful',
        });
      }
    } catch (error) {
      console.log(error);
      reply
        .code(500)
        .send({ message: error.response?.data || error.message || 'Connection failed' });
    }
  });

  // Get Parasol mode
  fastify.get('/parasol-mode', async (req: FastifyRequest, reply: FastifyReply) => {
    const parasolMode = getParasolMode();
    reply.send({ parasolMode });
  });
};
