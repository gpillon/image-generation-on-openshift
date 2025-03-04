import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { getLastActivity } from '../utils/config';

const routes = [
  {
    path: '/api',
    handler: async (_req: FastifyRequest, reply: FastifyReply) => {
      reply.send({ status: 'ok' });
    },
  },
  {
    path: '/api/kernels',
    handler: async (_req: FastifyRequest, reply: FastifyReply) => {
      reply.send([]);
    },
  },
  {
    path: '/api/terminals',
    handler: async (_req: FastifyRequest, reply: FastifyReply) => {
      reply.send(getLastActivity());
    },
  },
];

export default async (fastify: FastifyInstance): Promise<void> => {
  const prefix = process.env.NB_PREFIX || '';

  // Register all API routes
  routes.forEach(({ path, handler }) => {
    fastify.get(`${prefix}${path}`, handler);
  });

  // Add OpenShift AI specific routes if needed
  if (process.env.NB_PREFIX) {
    // Redirect all other prefixed requests to root
    fastify.get(`${prefix}/*`, async (_request: FastifyRequest, reply: FastifyReply) => {
      reply.redirect('/');
    });

    fastify.get(prefix, async (_request: FastifyRequest, reply: FastifyReply) => {
      reply.redirect('/');
    });
  }
};
