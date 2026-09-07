export class XMLHttpRequestMock {
  static instances: XMLHttpRequestMock[] = [];
  static nextStatus = 202;
  static nextResponseText = JSON.stringify({ status: 202 });

  method = "";
  url = "";
  body: Document | XMLHttpRequestBodyInit | null = null;
  headers: Record<string, string> = {};
  status = XMLHttpRequestMock.nextStatus;
  responseText = XMLHttpRequestMock.nextResponseText;
  upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor() {
    XMLHttpRequestMock.instances.push(this);
  }

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }

  send(body: Document | XMLHttpRequestBodyInit | null) {
    this.body = body;
    this.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 2 } as ProgressEvent);
    queueMicrotask(() => this.onload?.());
  }

  static reset() {
    XMLHttpRequestMock.instances = [];
    XMLHttpRequestMock.nextStatus = 202;
    XMLHttpRequestMock.nextResponseText = JSON.stringify({ status: 202 });
  }
}
