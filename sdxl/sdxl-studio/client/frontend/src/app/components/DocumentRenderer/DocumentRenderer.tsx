import { Spinner } from '@patternfly/react-core';
import * as React from 'react';

type DocumentRendererProps = {
    fileData: string;
    fileName: string;
    width: number;
    height: number;
};

const DocumentRenderer: React.FC<DocumentRendererProps> = ({ fileData, fileName, width, height }) => {
    if (fileName !== '') {
        const extension = fileName.split('.').pop()?.toLowerCase();
        const mimeType = extension === 'png' ? 'image/png' : `image/${extension}`;
        return <img src={`data:${mimeType};base64,${fileData}`} alt="image" width={width} height={height}/>;
    } else {
        return <div>
            <p style={{ textAlign: 'center' }}><Spinner size="xl" /></p>
            </div>;
    }
}

export default DocumentRenderer;