import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import fs from 'fs';
import path from 'path';

interface StudioConfig {
  disclaimer?: {
    status: string;
  };
}

export default async (fastify: FastifyInstance): Promise<void> => {
  fastify.get('/', async (req: FastifyRequest, reply: FastifyReply) => {
    try {
      const configFile = await fs.promises.readFile(
        '/opt/app-root/src/.local/share/sdxl-ministudio/config',
        'utf-8',
      );
      const studioConfig: StudioConfig = JSON.parse(configFile);
      const disclaimer = {
        status: studioConfig.disclaimer.status,
      };
      reply.send({ disclaimer });
    } catch (error) {
      const disclaimer = {
        status: 'unknown',
      };
      reply.send({ disclaimer });
    }
  });

  fastify.put('/', async (req: FastifyRequest, reply: FastifyReply) => {
    const { status } = req.body as any;
    const configFilePath = '/opt/app-root/src/.local/share/sdxl-ministudio/config';
    let studioConfig: StudioConfig = {};

    try {
      const configFile = await fs.promises.readFile(configFilePath, 'utf-8');
      studioConfig = JSON.parse(configFile);
    } catch (error) {
      if (error.code === 'ENOENT') {
        // File does not exist, initialize with an empty object
        studioConfig = {};
      } else {
        // Other errors
        reply.code(500).send({ message: 'Error reading config file', error });
        return;
      }
    }

    const disclaimer = {
      status: status,
    };

    studioConfig.disclaimer = disclaimer;

    try {
      await fs.promises.mkdir(path.dirname(configFilePath), { recursive: true });
      await fs.promises.writeFile(configFilePath, JSON.stringify(studioConfig, null, 2));
      reply.send({ message: 'Disclaimer status updated' });
    } catch (error) {
      reply.code(500).send({ message: 'Error writing config file', error });
    }
  });
};
