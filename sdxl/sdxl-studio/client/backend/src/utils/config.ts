// Initial configuration
let sdxlEndpointURL = process.env.SDXL_ENDPOINT_URL || '';
let sdxlEndpointToken = process.env.SDXL_ENDPOINT_TOKEN || '';
const parasolMode = process.env.PARASOL_MODE || 'false';

export const getSDXLEndpoint = (): any => {
  return {
    sdxlEndpointURL: sdxlEndpointURL.replace(/\/$/, ''),
    sdxlEndpointToken,
  };
};

export const setSDXLEndpoint = (url: string, token: string): void => {
  sdxlEndpointURL = url;
  sdxlEndpointToken = token;
};

export const getParasolMode = (): string => {
  return parasolMode;
};
