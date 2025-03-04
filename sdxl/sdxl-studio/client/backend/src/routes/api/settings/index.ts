import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';

import {
  getSDXLEndpoint,
  setSDXLEndpoint,
  getParasolMode,
  updateLastActivity,
} from '../../../utils/config';
import axios from 'axios';

export default async (fastify: FastifyInstance): Promise<void> => {
  // Retrieve endpoint settings
  fastify.get('/sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const endpointUrl = getSDXLEndpoint().sdxlEndpointURL;
    const endpointToken = getSDXLEndpoint().sdxlEndpointToken;
    const settings = {
      endpointUrl: endpointUrl,
      endpointToken: endpointToken,
    };
    console.log(settings);
    reply.send({ settings });
  });

  // Update endpoint settings
  fastify.put('/sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const { endpointUrl, endpointToken } = req.body as any;
    setSDXLEndpoint(endpointUrl, endpointToken);
    reply.send({ message: 'Settings updated successfully!' });
  });

  // Test endpoint connection
  fastify.post('/test-sdxl-endpoint', async (req: FastifyRequest, reply: FastifyReply) => {
    updateLastActivity();
    const { endpointUrl, endpointToken } = req.body as any;
    try {
      const response = await axios.get(endpointUrl + '/health?user_key=' + endpointToken);
      if (response.status === 200) {
        reply.send({
          message: 'Connection successful',
        });
      }
    } catch (error) {
      console.log(error);
      reply.code(500).send({ message: error.response.data });
    }
  });

  // Get Parasol mode
  fastify.get('/parasol-mode', async (req: FastifyRequest, reply: FastifyReply) => {
    const parasolMode = getParasolMode();
    reply.send({ parasolMode });
  });
};
